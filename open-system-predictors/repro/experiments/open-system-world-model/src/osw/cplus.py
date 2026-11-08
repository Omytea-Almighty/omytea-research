"""E3 training recipes (design: E3-DESIGN.zh.md §4, §10): classical CPLUS / CPLUS-M / SENS for GRU and LRU, and the
HQ recipes M (MoM warm start + fine-tune), C1 (compute-matched random restarts) and C2 (oracle start, diagnostic).

Shared protocol (§4): E2 hard-target multi-step NLL over k = 1..20 after a 32-step context, Adam, batch 64, gradient
clip 5, last checkpoint; validation NLL on the first 1,000 validation sequences; the test set is only predicted, never
used for selection. A checkpoint (model + Adam state) is kept every 100 steps; a non-finite loss or gradient stops the
phase, restores the last finite checkpoint and sets nan_flag. Init seeds follow the §4 namespace (init_seed below).

  r = cplus_single("LRU", 4, tr, va, te, 3, seed=200, replica=0, lrs=(3e-3, 1e-2))
  r = fit_M(tr, va, te, d=3, O=3, seed=200)

Every fit returns pred (n_test, 20, O; rows sum to 1 +- 1e-5), val_nll, n_params, cpu_s and JSON-serialisable
recipe diagnostics. The exact predictive is never an input here.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import hq_init, models, models_e2
from .run import predict_torch

L, K = 32, 20
BATCH, CLIP, CKPT_EVERY, N_VAL_SEL = 64, 5.0, 100, 1000
RECIPE = {"M": 0, "C1": 1, "S": 2, "C2": 3, "CPLUS": 4, "HMM-EM": 5, "OOM-MoM": 6, "HMM-MoM": 7, "SENS": 8}
NORM_TOL = 1e-5
MOD_MAX, MOD_MIN = 0.995, 1e-6


def init_seed(seed: int, recipe: int, r: int) -> int:
    assert 0 <= r < 1000 and 0 <= recipe < 10
    return 10 ** 7 + 10 ** 4 * seed + 10 ** 3 * recipe + r


def _setup():
    torch.set_num_threads(1)
    models.L_CTX = L


# ---------------------------------------------------------------- shared training machinery
def linear_lr(done: int, steps: int, lr0: float) -> float:
    """Learning rate after `done` of `steps` decayed steps: lr0 at the start, exactly 0 at the end."""
    return lr0 * (1 - done / steps)


def _tensors(tr):
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    return obs, act, obs[:, L:L + K]


def _nll(p, y):
    return -torch.log(torch.gather(p, 2, y.unsqueeze(-1)).clamp_min(1e-9)).mean()


def val_nll(model, va) -> float:
    pv = predict_torch(model, va[0][:N_VAL_SEL], va[1][:N_VAL_SEL], K)
    yv = va[0][:N_VAL_SEL, L:L + K]
    return float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())


def checked_pred(model, te) -> np.ndarray:
    pred = predict_torch(model, te[0], te[1], K).astype(np.float64)
    err = np.abs(pred.sum(-1) - 1).max()
    assert np.isfinite(pred).all() and err <= NORM_TOL, f"predictive not normalised: max |sum - 1| = {err}"
    return pred


def _snap(model, opt, step):
    return copy.deepcopy(model.state_dict()), copy.deepcopy(opt.state_dict()), step


def new_state(model, opt):
    return {"step": 0, "ckpt": _snap(model, opt, 0), "nan_flag": False, "nan_step": None, "curve": []}


def run_phase(model, opt, gen, data, lr0, steps, decay, st, on_ckpt=None) -> bool:
    """`steps` Adam steps at constant lr0 or linearly decayed to 0; returns False if stopped by a non-finite value.
    on_ckpt(step) runs after each checkpoint (read-only diagnostics; it draws no random numbers)."""
    obs, act, y = data
    for s in range(steps):
        lr = linear_lr(s, steps, lr0) if decay else lr0
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, obs.shape[0], (BATCH,), generator=gen)
        loss = _nll(model(obs[idx], act[idx], K), y[idx])
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if not (torch.isfinite(loss) and torch.isfinite(gn)):
            msd, osd, cstep = st["ckpt"]
            model.load_state_dict(msd)
            opt.load_state_dict(osd)
            st.update(nan_flag=True, nan_step=st["step"] + 1, step=cstep)
            return False
        opt.step()
        st["step"] += 1
        if st["step"] % CKPT_EVERY == 0 or s == steps - 1:
            st["ckpt"] = _snap(model, opt, st["step"])
            st["curve"].append((st["step"], float(loss.detach()), lr))
            if on_ckpt is not None:
                on_ckpt(st["step"])
    if decay:
        for g in opt.param_groups:
            g["lr"] = linear_lr(steps, steps, lr0)
    return True


def adam_step(opt) -> int:
    s = [v["step"] for v in opt.state.values() if "step" in v]
    return int(s[0]) if s else 0


def _diag(st, opt):
    return {"nan_flag": st["nan_flag"], "nan_step": st["nan_step"], "train_curve": st["curve"],
            "adam_step": adam_step(opt), "lr_end": float(opt.param_groups[0]["lr"])}


# ---------------------------------------------------------------- LRU spectral-lambda initialisation
def _word_freqs(obs, O, Lmax):
    """Pooled frequencies of every word of length 0..Lmax over all positions; words coded first-symbol-major."""
    obs = np.asarray(obs, dtype=np.int64)
    T = obs.shape[1]
    f, code = [np.ones(1)], np.zeros_like(obs)
    for n in range(1, Lmax + 1):
        code = code[:, :T - n + 1] * O + obs[:, n - 1:]
        f.append(np.bincount(code.ravel(), minlength=O ** n) / code.size)
    return f


def _words(O, maxlen=2):
    return [(n, c) for n in range(maxlen + 1) for c in range(O ** n)]


def hankel(obs, O, mid=None):
    """H[p, s] = P(p s) (mid=None) or P(p mid s), prefixes and suffixes = all words of length <= 2."""
    W = _words(O)
    f = _word_freqs(obs, O, 5)
    H = np.zeros((len(W), len(W)))
    for i, (lp, cp) in enumerate(W):
        for j, (ls, cs) in enumerate(W):
            if mid is None:
                H[i, j] = f[lp + ls][cp * O ** ls + cs]
            else:
                H[i, j] = f[lp + 1 + ls][(cp * O + mid) * O ** ls + cs]
    return H


def spectral_lambdas(train_obs, O) -> dict:
    """Data-driven rank r_hat and the non-stationary eigenvalues of S = sum_y A_y of a rank-r_hat spectral OOM."""
    train_obs = np.asarray(train_obs)
    H = hankel(train_obs, O)
    n = train_obs.shape[0]
    D = hankel(train_obs[:n // 2], O) - hankel(train_obs[n // 2:], O)
    thr = 3 * np.linalg.svd(D, compute_uv=False)[0] / np.sqrt(2)
    _, sv, Vt = np.linalg.svd(H)
    r = int(max(1, (sv > thr).sum()))
    V = Vt[:r].T
    P = np.linalg.pinv(H @ V)
    S = sum(P @ hankel(train_obs, O, mid=y) @ V for y in range(O))
    w = np.linalg.eigvals(S)
    w = np.delete(w, np.argmin(np.abs(w - 1)))
    w = w[w.imag >= 0]                                                    # one of each conjugate pair; reals once
    w = w[np.argsort(-np.abs(w), kind="stable")]
    return {"r_hat": r, "threshold": float(thr), "singular_values": [float(x) for x in sv],
            "lambdas": [[float(abs(x)), float(np.angle(x))] for x in w]}


def lru_spectral_lambda_init(model, train_obs, n_obs, info: dict | None = None) -> int:
    """Write nu = log(-log|lambda|), theta = arg(lambda) for the first min(n, count) spectral eigenvalues (modulus
    clipped to <= 0.995) into an LRUWM; the remaining entries keep their default init. Returns r_hat."""
    sp = spectral_lambdas(train_obs, n_obs)
    k = min(model.nu.shape[0], len(sp["lambdas"]))
    lam = np.array(sp["lambdas"][:k]).reshape(k, 2)
    mod = np.clip(lam[:, 0], MOD_MIN, MOD_MAX)
    with torch.no_grad():
        model.nu[:k] = torch.as_tensor(np.log(-np.log(mod)), dtype=model.nu.dtype)
        model.theta[:k] = torch.as_tensor(lam[:, 1], dtype=model.theta.dtype)
    if info is not None:
        info.update(sp, written=[[float(a), float(b)] for a, b in zip(mod, lam[:, 1])])
    return sp["r_hat"]


# ---------------------------------------------------------------- classical recipes
def cplus_single(family, size, tr, va, te, n_obs, seed, replica, lrs, steps_probe=750, steps_total=10000,
                 recipe=RECIPE["CPLUS"]) -> dict:
    """4 probes (2 inits x 2 lrs, constant lr), keep the lowest validation NLL, continue it with its own Adam state
    and batch generator for the remaining steps with lr decayed linearly to 0."""
    assert family in ("GRU", "LRU") and len(lrs) == 2
    _setup()
    t0 = time.process_time()
    data = _tensors(tr)
    probes, seeds, spec = [], [], {}
    for i in range(2):
        torch.manual_seed(init_seed(seed, recipe, 10 * replica + 2 * i))
        base = models_e2.build(family, size, n_obs, L)
        if family == "LRU" and i == 0:
            lru_spectral_lambda_init(base, tr[0], n_obs, info=spec)
        for j, lr in enumerate(lrs):
            ps = init_seed(seed, recipe, 10 * replica + 2 * i + j)
            m = copy.deepcopy(base)
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            gen = torch.Generator().manual_seed(ps)
            st = new_state(m, opt)
            run_phase(m, opt, gen, data, lr, steps_probe, False, st)
            probes.append((val_nll(m, va), m, opt, gen, st, lr))
            seeds.append(ps)
    probe_vals = [p[0] for p in probes]
    sel = int(np.argmin(probe_vals))
    _, m, opt, gen, st, lr = probes[sel]
    probe_nan = st["nan_flag"]
    continued = steps_total > steps_probe and not probe_nan        # §4: a NaN stops the run at its last finite checkpoint
    if continued:
        run_phase(m, opt, gen, data, lr, steps_total - steps_probe, True, st)
    res = {"pred": checked_pred(m, te), "val_nll": val_nll(m, va), "n_params": models.n_params(m),
           "cpu_s": time.process_time() - t0, "family": family, "size": size, "recipe": recipe, "replica": replica,
           "probe_seeds": seeds, "probe_lrs": [float(p[5]) for p in probes],
           "probe_val_nlls": probe_vals, "probe_nan_flags": [p[4]["nan_flag"] for p in probes],
           "selected_probe": sel, "selected_lr": float(lr), "selected_probe_nan": probe_nan, "continued": continued,
           "steps_probe": steps_probe, "steps_total": steps_total}
    res.update(_diag(st, opt))
    if family == "LRU":
        res.update(r_hat=spec["r_hat"], spectral_threshold=spec["threshold"], spectral_sv=spec["singular_values"],
                   spectral_lambdas=spec["lambdas"], spectral_written=spec["written"])
    return res


def cplus_m(family, size, tr, va, te, n_obs, seed, lrs, R_c, steps_probe=750, steps_total=10000,
            recipe=RECIPE["CPLUS"]) -> dict:
    """R_c independent CPLUS replicas; keep the lowest validation NLL. cpu_s is the total over replicas. Every
    replica's test prediction is kept in replica_preds (for D7's single- vs R_c-replica comparison; never selected on)."""
    best, vals, cpu, preds = None, [], 0.0, []
    for r in range(R_c):
        res = cplus_single(family, size, tr, va, te, n_obs, seed, r, lrs, steps_probe, steps_total, recipe)
        vals.append(res["val_nll"])
        preds.append(res["pred"])
        cpu += res["cpu_s"]
        if best is None or res["val_nll"] < best["val_nll"]:
            best = res
    out = dict(best, replica_val_nlls=vals, selected_replica=best["replica"], R_c=R_c, cpu_s=cpu,
               replica_cpu_s=best["cpu_s"], replica_preds=preds)
    return out


def sens(family, size, tr, va, te, n_obs, seed, lrs, R_c, base: dict | None = None, steps_probe=750,
         steps_total=10000) -> dict:
    """SENS: R_c extra replicas (recipe 8) merged with the existing CPLUS-M result `base` by validation NLL.
    cpu_s counts the extra replicas only; cpu_s_with_base adds the base job."""
    extra = cplus_m(family, size, tr, va, te, n_obs, seed, lrs, R_c, steps_probe, steps_total, RECIPE["SENS"])
    if base is None:
        return dict(extra, sens_source="extra", replica_val_nlls_extra=extra["replica_val_nlls"])
    best = base if base["val_nll"] <= extra["val_nll"] else extra
    return dict(best, sens_source="base" if best is base else "extra", cpu_s=extra["cpu_s"],
                cpu_s_with_base=extra["cpu_s"] + base["cpu_s"], replica_val_nlls_base=base["replica_val_nlls"],
                replica_val_nlls_extra=extra["replica_val_nlls"])


# ---------------------------------------------------------------- HQ recipes
def kraus_lists(m):
    Kr = m.kraus().detach().to(torch.complex128).numpy()
    return np.stack([Kr.real, Kr.imag], -1).tolist()


def hq_clamp_frac(m, obs) -> float:
    """Fraction of raw HQ predictive probabilities below the forward's clamp (models_e2.EPS) on these sequences."""
    with torch.no_grad():
        Ks, d = m.kraus(), m.d
        o = torch.as_tensor(obs)
        rho = (torch.eye(d, dtype=Ks.dtype) / d).expand(o.shape[0], d, d).clone()
        for t in range(L):
            Ky = Ks[o[:, t]]
            r = Ky @ rho @ Ky.conj().transpose(-1, -2)
            rho = r / torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.clamp_min(models_e2.EPS).to(Ks.dtype)[:, None, None]
        hit = tot = 0
        for _ in range(K):
            p = torch.einsum("yij,bjk,yik->by", Ks, rho, Ks.conj()).real
            hit, tot = hit + int((p < models_e2.EPS).sum()), tot + p.numel()
            rho = torch.einsum("yij,bjk,ylk->bil", Ks, rho, Ks.conj())
    return hit / tot


def _hq_train(m, tr, va, lr, steps, gen_seed, val_last=0):
    """Fine-tune with linear decay; val_last > 0 also logs the validation NLL at each checkpoint in the last val_last
    steps and its least-squares slope per step (G4's descriptive convergence check)."""
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(gen_seed)
    st = new_state(m, opt)
    st["val_curve"] = []

    def log(step):
        if step > steps - val_last:
            st["val_curve"].append((step, val_nll(m, va)))
    run_phase(m, opt, gen, _tensors(tr), lr, steps, True, st, log if val_last > 0 else None)
    return opt, st


def _val_slope(curve):
    if len(curve) < 2:
        return None
    x, y = np.array(curve, float).T
    return float(np.polyfit(x, y, 1)[0])


def _hq_result(m, va, te, opt, st, t0):
    res = {"pred": checked_pred(m, te), "val_nll": val_nll(m, va), "n_params": models.n_params(m),
           "cpu_s": time.process_time() - t0, "kraus": kraus_lists(m),
           "channel_spectrum": [list(x) for x in hq_init.channel_spectrum(m.kraus().detach())],
           "clamp_frac_test": hq_clamp_frac(m, te[0])}
    res.update(_diag(st, opt))
    if st.get("val_curve"):
        res.update(val_curve=st["val_curve"], val_slope_last=_val_slope(st["val_curve"]))
    return res


def fit_M(tr, va, te, d, O, seed, restarts=48, iters=150, steps=3000, lr=3e-3, val_last=0) -> dict:
    """M: MoM fit on training tables (selected by moment loss), warm start G = V P, fine-tune with linear decay."""
    _setup()
    t0 = time.process_time()
    torch.manual_seed(seed)
    tabs = hq_init.empirical_tables(tr[0], O)
    s0 = init_seed(seed, RECIPE["M"], 0)
    fit = hq_init.fit_mom(tabs, d, O, restarts=restarts, iters=iters, init_seed0=s0)
    m = models_e2.HQMMWM(O, d, L)
    hq_init.load_warm_start(m, fit["V"])
    pred_mom, init_val = checked_pred(m, te), val_nll(m, va)
    mom_spec = [list(x) for x in hq_init.channel_spectrum(m.kraus().detach())]
    opt, st = _hq_train(m, tr, va, lr, steps, seed, val_last)
    res = _hq_result(m, va, te, opt, st, t0)
    res.update(recipe=RECIPE["M"], init_seed0=s0, mom_loss=fit["mom_loss"], mom_losses_sorted=fit["losses_sorted"],
               selected_restart=fit["selected_restart"], mom_n_failed=fit["n_failed"], mom_cpu_s=fit["cpu_s"],
               restarts=restarts, iters=iters, steps=steps, lr=lr, init_val_nll=init_val, pred_mom=pred_mom,
               mom_channel_spectrum=mom_spec)
    return res


def fit_C1(tr, va, te, d, O, seed, R1, steps=6000, lr=1e-2) -> dict:
    """C1: R1 random HQ inits (recipe 1), each fine-tuned with linear decay; keep the lowest validation NLL."""
    _setup()
    t0 = time.process_time()
    best, vals, nans, seeds = None, [], [], []
    for r in range(R1):
        s = init_seed(seed, RECIPE["C1"], r)
        torch.manual_seed(s)
        m = models_e2.HQMMWM(O, d, L)
        opt, st = _hq_train(m, tr, va, lr, steps, s)
        v = val_nll(m, va)
        vals.append(v), nans.append(st["nan_flag"]), seeds.append(s)
        if best is None or v < best[0]:
            best = (v, r, m, opt, st)
    _, r, m, opt, st = best
    res = _hq_result(m, va, te, opt, st, t0)
    res.update(recipe=RECIPE["C1"], R1=R1, restart_val_nlls=vals, restart_nan_flags=nans, init_seeds=seeds,
               selected_restart=r, steps=steps, lr=lr)
    return res


def c2_start(m, V_true, sigma, noise_seed) -> torch.Tensor:
    """Write G0 = V_true + sigma (Z_re + i Z_im), Z ~ N(0, 1) per real component, directly into G (no P)."""
    V = torch.as_tensor(np.asarray(V_true), dtype=torch.complex128).reshape(m.n_obs * m.d, m.d)
    g = torch.Generator().manual_seed(noise_seed)
    zr = torch.randn(V.shape, generator=g, dtype=torch.float64)
    zi = torch.randn(V.shape, generator=g, dtype=torch.float64)
    G0 = V + sigma * torch.complex(zr, zi)
    with torch.no_grad():
        m.g_re.copy_(G0.real.to(m.g_re.dtype))
        m.g_im.copy_(G0.imag.to(m.g_im.dtype))
    return G0


def fit_C2(tr, va, te, d, O, seed, sigma, V_true, steps=3000, lr=3e-3, r=0, val_last=0) -> dict:
    """C2 (oracle diagnostic, never a capability arm): perturbed true isometry, then the same fine-tune as M."""
    _setup()
    t0 = time.process_time()
    torch.manual_seed(seed)
    m = models_e2.HQMMWM(O, d, L)
    ns = init_seed(seed, RECIPE["C2"], r)
    c2_start(m, V_true, sigma, ns)
    init_val = val_nll(m, va)
    opt, st = _hq_train(m, tr, va, lr, steps, seed, val_last)
    res = _hq_result(m, va, te, opt, st, t0)
    res.update(recipe=RECIPE["C2"], uses_oracle=True, sigma=float(sigma), noise_seed=ns, init_val_nll=init_val,
               steps=steps, lr=lr)
    return res
