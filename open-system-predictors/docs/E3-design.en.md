# E3 design: learnability repair for HQMMs on fresh seeds, tested at the true order — moment-matching warm start versus a Stiefel optimiser, with an oracle-start basin control, an E2-protocol reproduction control and a gated structure and parameter-efficiency block (revision 2) (English version)

Original: E3-DESIGN.zh.md (Chinese, not published). SHA-256 of the original: 69a5ad9cfe16a1dc7b016012e2c21fa19ff13b25baaf78263023a7c86bc94e11. The original is hash-committed: this hash carries RFC 3161 tokens from DigiCert (2026-10-24 15:57:50 UTC) and Sectigo (2026-10-24 15:57:51 UTC) (`repro/docs/timestamps/2026-10-24-e3-design/`). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

**Status:** draft, **not yet frozen**. [omitted: organisation or decision-maker reference] Before freezing, the development gates G1–G10 of §11 must be passed. At the freeze, the SHA-256 of this draft and of the code manifest are each computed and timestamped by DigiCert and Sectigo (same procedure as for v1.1 and E2), and the E3 prediction block is written into `PREDICTIONS-BEFORE-RESULTS.md`. No confirmatory job is run before these steps are complete. [Translator's note: this status line was written before the freeze and was not updated; the text translated here is the frozen version, whose SHA-256 and timestamps are given above.]
**The author's decisions on the open items raised by the design group (2026-10-24, before the freeze):**
1. **Compute cap:** G9's target becomes development plus confirmatory ≤ 28 CPU hours, hard cap 32 (previously 22 / 24). Reason: everything runs on local CPU at zero external cost; once the classical side is CPU-matched, the SC block is the most expensive part, and under today's machine load the original cap would very likely leave H6 unrun. The termination rule above 32 is unchanged (SC is not run within E3; any later run must follow the same frozen design, unchanged).
2. **H6 uses seed medians:** accepted. This is symmetric between HQ and the classical families and robust to a single plateau seed; the E2 comparison values in PC-K are still E2's means, and this is labelled as such.
3. **A3 and D4:** kept as descriptive items, not upgraded to a registered H1b. [Translator's note: in this design, D1–D8 denote the descriptive items listed in §6 under "Descriptive items (not adjudicated)"; they are distinct from the deviations D1–D16 of `DEVIATIONS.md`. "D15+" in §0 refers to a deviation entry.]
4. **Pre-freeze deletion rule for PC-A's median clause:** accepted (if the G2 count passes but the median is unreachable, the clause is deleted before the freeze and recorded in DEVIATIONS).
5. **Commit and timestamps:** freezing requires a git commit [omitted: internal administrative note] and submission of the SHA-256 of the design and of the code manifest to DigiCert and Sectigo (only the hashes are sent). **This step is taken only after confirmation by [omitted: organisation or decision-maker reference]**; before that, the implementation and the development gates may be completed, but nothing is frozen and no confirmatory job is run.

**The author's pre-freeze decisions, part two (after the implementation was complete and before the development gates were run, 2026-10-24; no development result had been seen at this point):** the implementation and two reviewers pointed out several situations whose handling the design did not state. They are settled as follows.
1. **Coverage simulation fails its criterion:** if BCa coverage < 0.94, switch to 97.5% intervals under the single registered fallback of §8, and report the measured coverage after the fallback; no second threshold is set.
2. **Planted-truth simulation fails its criterion:** as written in §9(ii), H6 is demoted to a descriptive item before the freeze; **the STOP rule is demoted to a descriptive item together with H6** (STOP depends on the same R_F machinery).
3. **Test bootstrap standard deviation of O05 on A1 and A2 > 0.01:** N_test for these two processes is fixed by the E2 protocol (C0 must reuse run_e2 bit for bit) and cannot be increased; if the bound is exceeded, this is reported as a limitation and **does not block the freeze**. For A2w and A3, N_test is still increased under §9(i).
4. **The A2 distinguishability check fails** (trivial or linear reference CF ≥ 0.85): H6 is demoted to a descriptive item before the freeze; this does not block the freeze.
5. **C0 is NaN at both learning rates and has no evaluable checkpoint:** impute in the direction unfavourable to the hypothesis; it is counted as NaN, not as a job failure.
6. **D1 and plateau labels:** the plateau-label definitions of revision 1 are not in this draft, and the definition of D1 is also incomplete; both are reported as "not computed", and only the channel spectrum is given.
7. **LRU spectral-λ initialisation:** remove the stationary eigenvalue closest to 1, then take min(n, r̂) eigenvalues (the stationary mode corresponds to the constant component and carries no decay information).
8. **The fine-tuning batch seeds of OOM-C and C2** are the run seed; **a failed C2 seed** is recorded as CF = 0 (this affects only the diagnostic arm).
9. **Freeze order:** the code-manifest SHA is computed only after the constants produced by the development gates have been written into `queue_e3.FROZEN_CONSTANTS`; the development records therefore appear stale after the freeze, but they are not re-run in the confirmatory phase.

**Development-gate results (G1–G10, development seeds 200–204, 2026-10-24; all PASS, `freeze_ok = True`; recorded in `runs/e3/GATES-DEV.json`):**
- G1: every job type runs through on seed 200; of the full test suite, 177 tests pass and 1 (the slow planted-truth test) is skipped.
- G2: on A2, O05 has CF ≥ 0.90 on 5/5 seeds, median 0.993; PC-A's median clause is **kept**; the number of fine-tuning steps stays at 3,000.
- G3: τ0 = 0.1 (lowest development validation NLL); R1 = 6, RS = 6 (about 108 and 114 CPU-s per restart respectively; 6 restarts bring the total CPU up to M's development median of 602 s).
- G4: development median of O05 on A2w is 0.991 ≥ 0.50; A2w is kept. G5: HMM4-C development CF 0.786–0.800 ≤ 0.88. G6: OOM3 3/3 ≥ 0.95; exact-Hankel fixture CF = 1. G7: A3 is descriptive only.
- **G8 (with changes):** all fixtures pass; A2 distinguishability: Markov maximum 0.097, linear context 0.771 < 0.85; O05 test bootstrap standard deviations all ≤ 0.0003, N_test unchanged. **Planted-truth simulation:** the false-call rate at R = 1 is 0.0 (meets the criterion); in the censored case with a true median of 0.92 at the largest rung, the mislabel rate is **0.287 > 2.5% (fails the criterion)**; the detection rate at R = 8 is 0.741 (insufficient power). Under §9(ii) and item 2 of the author's decisions, part two, **H6 and STOP are demoted to descriptive items before the freeze**. **Coverage simulation:** BCa coverage 0.882 < 0.94; under the §8 fallback, **the difference-type rules switch to 97.5% intervals**.
- **G9 (with changes):** projected from development-set measured CPU, development plus confirmatory is 34.8 CPU hours without trims; after all 7 trims in the registered order (O, D4, SENS, C2 σ = 0.2, A3, D3, GRU32/LRU32) it is **29.8 hours**, above the target of 28 and below the hard cap of 32, so the SC block runs as normal (the termination rule is not triggered). 310 confirmatory jobs.
- G10: for both GRU and LRU the two learning rates are {0.01, 0.03}; CPU_FLOOR = 602.4 s; R_c: GRU 2/4/8/16 is 5, GRU32 is 4, LRU 2/4/8 is 7, LRU 16/32 is 6; the PC-K(b) development check passes for GRU4 3/3 and LRU16 3/3; spectral rank on A2 r̂ = 3 (recorded only).
- The frozen constants have been written into `src/osw/queue_e3.FROZEN_CONSTANTS`; the registered, scored predictions are the E3 block of `PREDICTIONS-BEFORE-RESULTS.md` (revised by the author in light of the development results; the design group's numbers in §6 are kept as history).

**When it is run:** after the freeze. The code version is fixed during the confirmatory phase; each job records the code-manifest SHA when it starts.
**Nature:**
- An internal [omitted: internal project name] study, independent of the registered verdicts of v1/v1.1, E1 and E2; it changes none of them and does not reinterpret them.
- The results are not for external release. [Translator's note: a later decision publishes the results, together with this English version.] The "rule names" below are only the names of registered decision rules. Conclusions do not use expressions such as "quantum advantage", "first" or "beats".
- [omitted: internal project name]
- **Order convention:** all HQ and OOM rules are adjudicated at the **generator's true order**: on A1, HQ d=2 and OOM r=3; on A2, HQ d=3. Conclusions hold only under this convention. The case of a dimension above the true value (D4) is descriptive only.

---

## 0. Trigger: the two real failures that E3 addresses (compliance with [internal record] (the trigger rule))

**Failure 1: learnability of HQ on A2.** The frozen E2 results show that on A2, the weakly measured qutrit clock, the validation NLL of HQ(3) stalls near ln 3, with a mean CF of 0.14 over 5 seeds. Of the 10 runs, 7 have CF ≤ 0.048 (plateau P1) and 3 lie at 0.206–0.218. Yet this model family contains the true model: with the true Kraus operators loaded, the difference from the exact solution is below 2×10⁻³.

**Failure 2: the positive control was not calibrated.** The budgets of the A1 positive control (HQ against HMM) were set at 64 and 256 parameters. The known effect appears below about 24 parameters; at these two budgets the HMM has essentially caught up, so the threshold 0.10 is unreachable for the positive control itself. Under the E2 design, the registered main conclusion is therefore not accepted ([omitted: private document], correction after review round 1). [Translator's note: this outcome is stated in the technical report, §4.3, "Validity of the registered conclusion".] In addition, per the correction of 2026-10-24 in [omitted: private document]: the predictable information of A2 is 0.0180 nats (k ≤ 10) and 0.0128 nats (k ≤ 20); the value of 0.006 written earlier is void.

**Compliance with [internal record] (the trigger rule).** The rule is "no new theory line without a ledger failure". Each arm of E3 follows directly from the two registered failures above:
- A1 and A2 are E2's original processes;
- A2w directly calls the existing function `tasks_e2.qutrit_clock(eps=0.15)`;
- A3 (d=4) is another instance of the same construction rule and is not triggered by any registered failure, so it is **demoted to the purely descriptive item D8** and carries no decision rule. The reason for keeping it (to examine how the repair behaves as the dimension grows) is written into DEVIATIONS.md as a D15+ entry and registered before the freeze; [Translator's note: this entry was not written before the freeze; deviation D15 in `DEVIATIONS.md` records the omission and registers the reason after the freeze.]
- OOM, HMM, GRU and LRU serve only as comparison objects and introduce no new generator.

**Exploratory evidence (2026-10-24; all on exploratory seeds ≥ 300; adversarially re-checked, and only what survived the re-check is trusted):**
1. **MoM repair.** [Translator's note: MoM = method of moments; the MoM warm start is the moment-matching warm start of the title and of recipe M (§4), the term used in the technical report.] MoM warm start plus fine-tuning with linear decay gives CF 0.992–0.995 on 7 new exploratory seeds; two independent implementations agree. Comparisons:
   - the same fine-tuning started from random initialisation: CF 0.012, −0.013, 0.036;
   - the unchanged E2 protocol: 0.03–0.22 (re-check seeds 0.051, 0.028, 0.153);
   - moment loss of the selected fit 0.0004–0.0006; reference moment loss of the true model 0.00017–0.00023.
2. **Oracle start.** σ·N(0,1) noise is added to each real component and written directly into G (`theory/oracle2.py`, lines 20–21): [Translator's note: `theory/oracle2.py` was an exploratory script that was not retained and is not part of this supplement; the registered noise convention is stated in full under recipe C2 in §4.]
   - σ ≤ 0.1: CF 0.91–0.97 (3 noise seeds, constant lr, 2,000 steps);
   - σ = 0.2: varies with the seed, 0.43 / 0.026 / 0.03;
   - σ = 0.3: 0.005 / 0.004 / 0.13.
   This shows that a data size of N = 20k is sufficient.
3. **Polar-decomposition trap.** An eigh-based polar decomposition has a NaN backward gradient at an exactly isometric G, so the warm start writes G = V·P, with P a non-degenerate positive-definite diagonal matrix.
4. **Reachability of the positive control.** The best HMM(4) obtained by fitting A1's exact length-10 word distribution with EM has CF 0.800–0.805 under the E2 protocol (independently reproduced, 20 restarts).
5. **Theory (re-checked):**
   - HQ(d) ⊆ OOMs of dimension ≤ d² (Adhikary et al. 2020);
   - Hankel rank: 3 for A1, 9 for A2; A1 has no finite HMM;
   - the CF ceiling of the softmax linear-context predictor on A2 is 0.768, and 0.907 at ε = 0.15;
   - the unconstrained spectral OOM on A2 gives many negative predictions at N = 20,000–200,000.
6. **Compute measurements (this revision):** `hq_init.fit_mom` timed with `process_time`, machine load 10–25:
   - A2 (d=3, O=3): 10.6–13.1 CPU-s per restart;
   - A3 type (d=4, O=4): 11.9 CPU-s per restart, a ratio of about 1.0–1.15 to d=3;
   - the re-check implementation at low load [omitted: internal administrative note]: 48 restarts take 204–259 CPU-s in total, 258–319 CPU-s including fine-tuning.
   Load inflates `process_time` by about a factor of 2, so §13 uses low-load estimates and G9 re-projects with development-set measurements. Scripts: [omitted: private file path].
7. **Not reproduced, not registered:** the "P2 geometric trap" and "two-level plateau" mechanisms, the minibatch signal-to-noise numbers, the threshold "moment loss must be < 0.001", and the individual single-setting negative results.

---

## 1. Literature check (limited search scope; no claim of exhaustiveness)

- **Srinivasan, Gordon, Boots (AISTATS 2018).** On the probability clock, an 8-parameter HQMM needs an HMM of about 80 parameters to be matched; the paper states that its optimisation is prone to local optima.
- **Adhikary et al. (AISTATS 2020, COSM).** HQMMs are a subclass of OOMs; whether the subclass is proper remains open. The optimiser is the Wen–Yin Cayley retraction on the Stiefel manifold: momentum 0.9, one gradient normalisation before and one after the momentum step, τ ∈ [0.55, 0.95]. This experiment's S arm re-implements it accordingly.
- **NS-RIS (Ning 2026) and SHQMM (Li 2024).** The reported DA is higher than the DA of the true generator itself; the results are not internally credible and are not used as evidence.
- **Banchi 2024, Elliott et al. PRL 2020, Anschuetz et al. 2023, Wang et al. 2026.** These have proven classical-memory separations. Including them would need new generators; E3 does not include them.
- **Miller et al. 2021.** Only weak evidence that "Adam itself is not the cause".
- **Gap (within the search scope):** no one has compared HQMM-type models with GRU/LRU on a small-parameter ladder against an exact predictive reference; nor is there any literature record of a learnability repair for HQMMs on weak-signal realisable processes.

---

## 2. Process definitions (all start from the stationary state; context 32 steps; prediction k = 1…20)

| Code | Definition | d / O | Relaxation (steps) | Predictable information D (k ≤ 10, nats) | Hankel rank | Linear-context ceiling | Markov-1/2/3 CF (exploratory) | Train / validation / test | Data code | Role |
|---|---|---|---|---|---|---|---|---|---|---|
| A1 | `tasks_e2.probability_clock()` | 2 / 2 | 2.71 | about 0.019 | 3 | 0.592 | 0.33 / 0.29 / 0.68 | 20k / 2k / 2k | 1 | H3a, PC-B, PC-K(b), H7 |
| A2 | `tasks_e2.qutrit_clock()`, ε = 0.3271, θ = 0.6 | 3 / 3 | 20.0 | 0.0180 | 9 | 0.768 | 0.05–0.20 | 20k / 2k / 2k | 2 | main question |
| A2w | `tasks_e2.qutrit_clock(eps=0.15)` | 3 / 3 | 93.8 | 0.0031 | 9 | 0.907 | ≤ 0.04 | 120k / 2k / 10k | 3 | H3b |
| A3 | `tasks_e3.qudit_clock(d=4, eps=0.2812632554285184, θ=0.6)` | 4 / 4 | 20.000 | 0.0177 (exploratory value) | ≤ 16 | computed at the development gate | 0.04 / 0.06 / 0.09 | 40k / 2k / 4k | 4 | descriptive only (D8) |

- **Definition of A3:**
  - F is the 4-point Fourier matrix, H = F·diag(0, 1, √2, √3)·F†, U = exp(−i·0.6·H);
  - E_y = (1−ε)·I/4 + ε|y⟩⟨y|, K_y = √E_y·U;
  - ε is set by bisection so that the relaxation time equals 20.000 steps;
  - at d = 3 the construction agrees bit for bit with `qutrit_clock`, which a test guarantees.
- **Number of training sequences (a heuristic rule, registered as constants):**
  - D is required to be at least 10 times p_intr/(2N), where p_intr = 2Od² − 2d² + 1 − O.
  - This is an order-of-magnitude heuristic: D is a mean KL per step, whereas the Wilks quantity is an excess loss over the whole likelihood, and the two have different units. It therefore **does not constitute** an argument that "identifiability has been matched", and that statement is not used in the H3b conclusion either.
  - Values: A1 and A2 keep 20k, so that C0 can reuse the E2 protocol bit for bit; A2w is 120k; A3 is 40k.
- **Data generation:**
  - Each (process, seed) uses `np.random.default_rng(10000·seed + data code)` and calls `tasks_e2.sample` in the order training, validation, test. The data codes of A1 and A2 are the same as in `run_e2.data`.
  - The exact predictive distributions of the test set are generated together with the data, but training and selection code **must not read** them. This is enforced by the poisoning test of §9(iii).

---

## 3. Primary metric and reported quantities

- **Primary metric:** CF = 1 − mean_{k≤10} KL(exact‖model) / mean_{k≤10} KL(exact‖marginal). The code imports directly:
  - CF itself from `queue_e2.metric` (`src/osw/queue_e2.py`, lines 62–66);
  - KL from `run.metrics` (p clipped at 1e-9, q clipped at 1e-12);
  - `kl_marginal` copies `run_e2.main` lines 126–128 verbatim: the marginal is `exact[:, -1].mean(0)`.
- **Normalisation assertion:** for every family, the predictive distribution must sum to within 1 ± 1e-5 on every (sequence, k); this is checked **after** any clipping or renormalisation and **before** any metric is computed. A unit test shows that the assertion is necessary: 1.1 × the exact distribution gives CF > 1.
- **OOM evaluation variant (registered):** the raw output is first passed through `clamp_min(p, 1e-6)` and renormalised (as in the training loss); then the assertion is checked and CF is computed. Also reported: the negative-prediction rate NR, CF computed only on valid pairs, and raw-output CF (labelled as unnormalised). H7, PC-O and D2 all use this variant.
- **Quantities reported only, not adjudicated:**
  - validation NLL; KL per k;
  - the moment loss of the selected fit and all 48 sorted moment losses; CF of the MoM fit alone;
  - the learned channel spectrum and plateau labels (definitions as in revision 1: P1, escaped, P2);
  - `cpu_s` of each job (`process_time`; for the C0 subprocess, `getrusage(RUSAGE_CHILDREN)`) and `wall_s`;
  - the number of NaN events; the rate at which HQ triggers `clamp_min`;
  - the one-sided 95% Clopper–Pearson bound for each count rule.

---

## 4. Frozen training recipes

**Shared settings:**
- E2 loss: hard targets, multi-step NLL, k = 1…20.
- Adam, batch 64, gradient clipping 5, last checkpoint taken.
- No selection uses the test set; validation NLL always uses the first 1,000 validation sequences.
- **Batch-generator seed:**
  - for C0 and M, equal to the run seed (e.g. 20), consistent with E2's `train()`: `torch.Generator().manual_seed(a.seed)`;
  - for other recipes with several restarts or probes, equal to the initialisation seed of that restart or probe.
- **Initialisation seed namespace:** init = 10⁷ + 10⁴·seed + 10³·recipe code + r.
  - Recipe codes: M-MoM 0, C1 1, S 2, C2 3, CPLUS 4 (r = 10·replica + probe), HMM-EM 5, OOM-MoM 6, HMM-MoM 7, SENS 8.
  - Confirmatory seeds fall in 10,200,000–10,299,999 and development seeds in 12,000,000–12,099,999, disjoint from the seeds of E2, v1.1 and the exploratory scripts (tested).
- **NaN handling:** a checkpoint is saved every 100 steps, identically for all arms. On a NaN gradient, training stops immediately, the last finite checkpoint is evaluated, and `nan_flag` is set. This does not count as a job failure.
- **Threads:** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `OPENBLAS_NUM_THREADS` are all set to 1, and `torch.set_num_threads(1)` is called. There is a bit-for-bit rerun test.

**Recipes:**

- **M (MoM warm start plus fine-tuning; the HQ recipe under test, not tuned after the freeze)**
  1. Compute moment tables from the training sequences only: pair tables for k = 1…10, triple tables for (k1, k2) ∈ {1,2,3}² (`hq_init.empirical_tables`).
  2. Compute the model tables exactly from the model's stationary state (`hq_init.model_tables`). The loss is the sum of the KLs over the tables.
  3. 48 random restarts: each real component of G drawn from N(0, 0.5²). L-BFGS: lr 1, history 20, strong-Wolfe, at most 150 iterations. Selection by training moment loss only.
  4. Write G = V_sel·diag(linspace(1, 1.6, d)).
  5. Fine-tuning: lr 3e-3, 3,000 steps, linear decay to 0.

- **C0 (the unchanged E2 protocol)**
  - The unmodified `run_e2.main` is called in a subprocess, **and `--out runs/e3/c0` must be passed**;
  - lr 3e-3 and 1e-2, each 6,000 steps at constant lr; the one with the lower validation NLL is kept;
  - subprocess CPU is recorded with `getrusage`.

- **C1 (compute-matched random restarts plus decay)**
  - R1 random initialisations, each lr 1e-2, 6,000 steps, linear decay to 0, selected by validation NLL;
  - R1 ∈ {3, 4, 5, 6} is set at G3 to the smallest value whose development-set CPU ≥ M's development-set median CPU, and frozen.

- **S (COSM-style Stiefel optimiser)**
  - V is a parameter directly on St(O·d, d); Wirtinger gradients, normalised once before and once after momentum 0.9;
  - Cayley update, τ_t = τ0·0.9^{t/313}, with τ0 chosen at G3 from {0.1, 0.3, 0.6, 0.9};
  - RS restarts, 6,000 steps each, selected by validation NLL. RS is set in the same way as R1.

- **C2 (oracle start; diagnostic only, never enters a capability comparison)**
  - **Noise convention (registered):** G0 = V_true + σ·(Z_re + i·Z_im), where every entry of Z_re and Z_im is independently N(0,1), i.e. each real component has variance σ². Same as `theory/oracle2.py`, lines 20–21.
  - G0 is written directly into G, without multiplying by P (G0 is generally non-degenerate), followed by the same fine-tuning as M.
  - A2 and A2w use σ = 0.05. A3 uses σ_A3 = 0.0375, so that the relative perturbation σ·√(2Od) equals that of A2.

- **CPLUS (single-replica recipe for GRU/LRU)**
  - 4 probes: 2 initialisations × 2 lr (the two values selected by the G10 learning-rate grid), each 750 steps at constant lr;
  - one probe is selected by validation NLL and continued, with its Adam state, for 9,250 steps with lr decaying linearly to 0.
  - One of the two LRU initialisations is the **spectral-λ initialisation**:
    - the rank is no longer hard-coded as 9; instead, r̂ = the number of singular values of the training Hankel matrix (prefixes and suffixes are words of length ≤ 2) above the noise threshold;
    - noise threshold = 3 × (largest singular value of the difference between the Hankel matrices of the two halves of the training set / √2);
    - take min(n, r̂) eigenvalues, only one from each conjugate pair, with moduli truncated to ≤ 0.995, and set ν = log(−log|λ|), θ = arg λ;
    - r̂ is recorded.

- **CPLUS-M (compute-matched replicas; used for the decisive ladder)**
  - Run R_c(F, size) independent CPLUS replicas and take the one with the lowest validation NLL;
  - R_c = ceil(CPU_FLOOR / development-set CPU of a single replica at that size), with CPU_FLOOR = the development-set median `cpu_s` of HQ3-M on A2;
  - both are determined at G10 and frozen as constants.

- **SENS (robustness check on the classical side)**
  - On each family's "lower bracketing rung", run another R_c replicas, seeds 20–24, pool them with the existing replicas and select by validation NLL. Total compute is about 2× CPU_FLOOR.
  - The "lower bracketing rung" is the rung immediately before the one at which the upper envelope on CPLUS first crosses 0.90; if right-censored, the largest rung.

- **HMM-C**
  - Softmax-parameterised T and B, with π the stationary distribution, fitted to the same moment tables: 48 restarts, L-BFGS 150 iterations, selected by moment loss;
  - then Baum–Welch on all 20k training sequences, ≤ 100 iterations, relative tolerance 1e-6;
  - in addition, 3 randomly initialised EM runs; the best by training log-likelihood is kept, and all fits are reported.

- **OOM-C**
  - σ = e1 gauge; the first row of A_O is constrained so that Σ_y p = 1 holds exactly. Free parameters: O·r² − 1.
  - Two initialisations: spectral initialisation and MoM initialisation (48 restarts). Each is fine-tuned for 3,000 steps, lr 3e-3, linear decay.
  - **The candidate set includes the spectrally initialised model before fine-tuning**; selection by validation NLL. This can only favour the OOM.

- **REF**
  - Markov-1/2/3 from training counts (add-½ smoothing);
  - the linear-context ceiling;
  - the context-shuffled exact predictor.

---

## 5. Arms, seeds and jobs

**Seeds:** confirmatory seeds 20–29. The development gates use development seeds 200–204; 205–209 are reserved for a single re-run of the gates after a code fix. None of these ranges has been used before: E2 used 0–4, v1.1 used 0–9 and 100–109, exploration uses ≥ 300.

| Block | Arm | Family and size (parameter count) | Recipe | Seeds | Jobs |
|---|---|---|---|---|---|
| U | L-A2-M (H1) | HQ(3) (54) | M | 20–29 | 10 |
| U | L-A2-C0 (PC-C) | HQ(3) | C0, 2 lrs | 20–29 | 20 |
| U | L-A2-C1 (H5) | HQ(3) | C1 | 20–29 | 10 |
| U | L-A2-S (H2) | HQ(3), Stiefel | S | 20–29 | 10 |
| U | L-A2-C2 (PC-A, H4) | HQ(3) | C2, σ ∈ {0.05, 0.1, 0.2, 0.3} | 20–29 | 40 |
| U | L-A1-M / C0 / C2 (H3a, PC-B, H7) | HQ(2) (16) | M; C0 (2 lrs); C2, σ = 0.05 | 20–29 | 10 + 20 + 10 |
| U | PC-B | HMM n ∈ {3, 4} (11, 19) | HMM-C | 20–29 | 20 |
| U | PC-O / H7 | OOM(3) (17), A1 | OOM-C | 20–29 | 10 |
| U | REF and fixtures | 4 processes | REF | 20–29 | 40 |
| T (H1 is PASS or PARTIAL) | A2w-M / C2 (H3b) | HQ(3) | M; C2, σ = 0.05 | 20–29 | 10 + 10 |
| T | A3-M / C2 (D8, descriptive) | HQ(4) (128) | M; C2, σ = 0.0375 | 20–24 | 5 + 5 |
| T | D4 (descriptive) | HQ(4) (96), A2 | M | 20–22 | 3 |
| SC (H1 PASS and PC-B PASS) | A2 decisive ladder | GRU h ∈ {2, 4, 8} (63, 147, 387); LRU n ∈ {2, 4, 8} (59, 99, 259) | CPLUS-M | 20–29 | 60 |
| SC | A2 upper ladder | GRU h ∈ {16, 32} (1155, 3843); LRU n ∈ {16, 32} (771, 2563) | CPLUS | 20–29 | 40 |
| SC | PC-K(b) | GRU4 (130), LRU16 (722) on A1 | CPLUS | 20–29 | 20 |
| SC | SENS (D7) | lower bracketing rung of each family | SENS | 20–24 | 10 |
| SC | D3 (descriptive) | LRU16 on A2w, using only the first 20,000 training sequences | CPLUS | 20–24 | 5 |
| O (PC-O PASS) | D2 (descriptive) | OOM(9) on A2 (242, true rank) | OOM-C | 20–29 | 10 |

- 378 confirmatory jobs; 84 development jobs plus fixtures. [Translator's note: the development-gate results at the head of this document record that all seven registered compute trims of G9 (§11) were applied before the freeze (O, D4, SENS, C2 σ = 0.2, A3, D3, GRU32/LRU32), leaving 310 confirmatory jobs; 378 is the count before these trims.]
- Definition of the decisive ladder: whether H6 holds (P_F ≥ 216, i.e. 4×54) and the kill line (P_F < 108) depend only on whether the upper envelope crosses 0.90 at GRU ≤ 387 and LRU ≤ 259. The upper envelope is a cumulative maximum, so larger rungs cannot move the crossing point down. Compute matching is therefore applied only to these three rungs; the upper ladder affects only the censoring bound.
- The HMM is not re-run on A2: a fit to the exact word distribution gives CF ≈ 0.20 (0.2033) at 287 parameters. This is an exploratory value and a conjecture, and is cited only descriptively.

---

## 6. Hypotheses and decision rules (fixed in advance; every rule is an **exhaustive and mutually exclusive** ordered table, executed mechanically by the frozen summary code, with unit tests enumerating the full input grid)

[Translator's note: the prediction probabilities given in §6 and §7 were written before the development results were seen and are kept as history; the registered, scored E3 predictions are the E3 block of `PREDICTIONS-BEFORE-RESULTS.md`. Under the development-gate results at the head of this document, the difference-type rules (H5, H7, PC-B) use 97.5% instead of 95% intervals (§8 fallback), and H6 and the STOP rule were demoted to descriptive items before the freeze.]

All count rules are count thresholds only; no p-values are reported. "≥" carries a 1e-9 relative tolerance, and an exact tie is treated as unfavourable to the hypothesis. Every count conclusion is reported together with the one-sided 95% Clopper–Pearson bound: the lower bound is 0.61 for 9/10 and 0.74 for 10/10; the upper bound is 0.39 for 1/10 and 0.61 for 3/10.

### Family 1 (main question)

**H1 (MoM repair on A2, true order d=3)**
- A seed is a "success" if CF_test(k ≤ 10) ≥ 0.90. Let s = the number of successes among seeds 20–29.
- Ordered table, evaluated in order:
  1. PC-A fails → **UNINTERPRETABLE**;
  2. s ≤ 6 → **KILL**;
  3. s ≥ 9 → **PASS**;
  4. otherwise, i.e. s = 7–8 → **PARTIAL**.
- The mean, median and CP lower bound are reported only and **do not enter the label**. This removes the defects of revision 1, in which "9/10 but mean < 0.90" had no label and "8/10 and mean < 0.80" had two labels.
- Conclusion wording: PASS is written as "reached the registered 9/10 threshold (95% CP lower bound on the success rate 0.61 or 0.74)", not as "reliable".
- Prediction: PASS 0.70, PARTIAL 0.20, KILL 0.10.

**H2 (Stiefel optimiser, starting from random initialisation)**
- A seed is a "success" if the restart with the lowest validation NLL has CF ≥ 0.90.
- Ordered table:
  1. s ≥ 9 → "S reached the threshold";
  2. s ≤ 1 → "S did not reach the threshold" (95% CP upper bound on the success rate 0.39);
  3. otherwise → **PARTIAL**.
- Conclusion wording: "under this implementation, τ0 grid and compute, S reached / did not reach the threshold", not "cannot". S and M differ simultaneously in parameterisation, gradient normalisation, schedule and restarts, so H2 does not support a causal attribution concerning the cause of the E2 failure; "the cause is Adam plus the polar decomposition" is reported only as a hypothesis.
- Prediction: not reached 0.55, PARTIAL 0.35, reached 0.10.

### Family 2 (transfer)

**Per-seed success definition (common to H3a and H3b):** let O05 denote the C2 result on the same seed.
- If CF(O05) < 0.50, the seed counts as unsuccessful for M (reference failure);
- otherwise, success = CF(M) ≥ 0.95·CF(O05) **and** CF(M) ≥ 0.50.

**H3a (A1, no-harm test; always adjudicated, whatever the outcome of H1)**
- Success additionally requires CF(M) ≥ CF(C0) − 0.01.
- Ordered table: ≤ 5 → KILL; ≥ 9 → PASS; 6–8 → PARTIAL.
- Prediction: PASS 0.85, PARTIAL 0.12, KILL 0.03.

**H3b (A2w, weak signal; adjudicated by the rule only if H1 is PASS; if H1 is PARTIAL, descriptive only)**
- Precondition: G4 passed, and the median CF of O05 on the confirmatory seeds ≥ 0.50; otherwise the label is "statistically limited, not judged".
- Ordered table as for H3a (without the C0 condition).
- Meaning of KILL: at the registered N, the repair does not get past the hurdle of "weak per-step signal, memory of about 94 steps".
- Prediction: PASS 0.45, PARTIAL 0.30, KILL 0.20, not judged 0.05.

### Mechanism and attribution (fixed thresholds, always reported)

**H4 (basin locality)**
- s(σ) = the fraction of the 10 seeds with CF ≥ 0.90. s(0.05) is the same data as PC-A and carries no separate conclusion.
- Ordered table: s(0.3) ≤ 0.2 → "local basin"; s(0.3) ≥ 0.6 → "non-local"; otherwise → "undetermined".
- The interpolated radius σ* is also reported.
- Prediction: local 0.70, undetermined 0.20, non-local 0.10.

**H5 (attribution to the warm start)**
- Statistic Δ5 = mean CF(M) − max(mean CF(C0), mean CF(C1)), with a BCa 95% CI (see §8).
- Ordered table:
  1. mean CF(C1) ≥ 0.80 → "refuted (compute plus decay already suffices)";
  2. CI lower bound ≥ 0.40 and mean CF(C1) ≤ 0.50 → "established";
  3. otherwise → "attribution undetermined".
- **NaN robustness:** computed in two versions, one including the runs with `nan_flag` and one with those runs removed; "established" requires both versions to satisfy the condition.
- Prediction: established 0.80, undetermined 0.15, refuted 0.05.

### Family 3 (structure and parameter efficiency; α = 0.025 per rule)

**H6, rule name "A2-P90 ratio"**

[Translator's note: H6 and the STOP rule below were demoted to descriptive items before the freeze, because the planted-truth simulation of §9(ii) failed its criterion at the development gate (mislabel rate in the censored case 0.287 > 2.5%); see the development-gate results at the head of this document.]

- **Gating:** H1 PASS, PC-B PASS, PC-K PASS, compute check passed. If any of these is not met → UNINTERPRETABLE or "not judged", and the reason is reported.
- **Registered statement:** on A2, with HQ at its true order d=3, for both GRU and LRU: the point estimate of P_F/54 ≥ 4, **and** the one-sided lower bound ≥ 2.
- **Curve statistic:** all families uniformly use the **seed-median CF**, symmetric between HQ and the classical families. The median is not affected by a single plateau seed; when H1 is PASS, the HQ median is necessarily ≥ 0.90.
- **Computing P_F:**
  - take the monotone upper envelope (cumulative maximum) of the median CF along the ladder, and interpolate linearly in log(parameter count) between the two adjacent rungs between which it first crosses 0.90;
  - **left-censoring:** if the smallest rung is already ≥ 0.90, P_F = that rung's parameter count (R ≤ 1.17), which triggers the kill line;
  - **right-censoring:** if it never crosses, P_F = P_max.
- **Bootstrap:** 10,000 resamples, resampling the same set of seeds jointly for HQ and the classical families. If the HQ median in a resample is < 0.90, that resample is recorded as R = 0. The one-sided lower bound is the **1st percentile**; a percentile more conservative than the nominal 2.5% is used to compensate for the under-coverage of the percentile bootstrap at n = 10, and is calibrated by the planted-truth simulation of §9.
- **Ordered table (per family):**
  1. point estimate R_F < 2 → **KILL** (and triggers the STOP check);
  2. point estimate ≥ 4 and lower bound ≥ 2 → component holds;
  3. point estimate ≥ 4 and lower bound < 2 → "not established (imprecise)";
  4. 2 ≤ point estimate < 4 → "not established".
- The rule holds if and only if both components hold.
- **Reporting a censored component:** if ≥ 50% of the resamples are censored, it is reported as "median CF 0.90 not reached within ≤ P_max parameters (median CF at the largest rung = X)", not as a parameter ratio, and without a p-value.
- **Note on LRU:** if LRU is censored and the CF at the largest rung lies within ±0.03 of the linear-context ceiling 0.768, the annotation "possibly an architectural ceiling rather than parameter efficiency" is added.
- **Robustness (D7, registered):** on seeds 20–24, replace the values at the lower bracketing rung with SENS, compare with the CPLUS-M results on the same 5 seeds, and recompute the R_F point estimate. If either family drops from ≥ 4 to < 4, flag "H6 not robust"; if ΔCF ≥ 0.03, flag "classical curve limited by optimisation". Both flags are reported together with H6 and **do not change** the registered label.
- Prediction: holds 0.65 conditional on the gates passing; about 0.40 unconditionally.

**H7, rule name "A1-positivity equivalence"**
- **Gating:** PC-O PASS and H3a not KILL. H3a is always adjudicated, so this is independent of H1.
- Both families at their true order: HQ(2)-M has 16 parameters, OOM(3)-C has 17 parameters; the OOM uses the clipped-then-renormalised variant registered in §3.
- Δ7 = mean[CF(HQ2-M) − CF(OOM3)], with a BCa 95% CI.
- Ordered table:
  1. CI ⊂ [−0.03, +0.03] → "equivalent" (TOST, 0.025 per side);
  2. Δ7 ≥ 0.05 and CI lower bound > 0 → "HQ better under these training recipes"; if the OOM's CF after fine-tuning is below the CF of its spectral initialisation, relabel as "OOM optimisation failure", with no structural interpretation;
  3. Δ7 ≤ −0.05 and CI upper bound < 0 → "OOM better";
  4. otherwise → "uncertain".
- Prediction: equivalent 0.70, HQ better 0.08, OOM better 0.05, uncertain 0.17.

### STOP rule

- **Trigger:** all gates of H6 pass, and the point estimate min_F R_F < 2, with **CPLUS-M as the only input**; SENS does not enter STOP.
- **Registered outcome:** close the sub-direction "parameter efficiency of HQ relative to analogue-state RNNs, on weak-signal processes generated by HQ itself, at the true order". Reopening it requires a trigger from a new real failure.
- Prediction: triggered 0.07 conditional on the gates passing.

### Descriptive items (not adjudicated)

- **D1:** small-budget differences on A2, descriptive only.
- **D2:** NR of OOM(9) on A2, and the CF of each evaluation variant.
- **D3:** LRU16 on A2w compared with the linear ceiling. **Summary tables must not place HQ (120k) and LRU (20k) on A2w side by side.**
- **D4:** results of HQ(4)-M on A2 (behaviour when the dimension exceeds the true value), 3 seeds.
- **D5:** moment loss, plateau labels, channel spectrum.
- **D6:** CPU of each arm, and the CPU ratio per cell.
- **D7:** robustness of H6 (see above), and the ΔCF between a single replica and R_c replicas.
- **D8:** success fraction of M relative to O05 on A3 (d=4) (5 seeds).

---

## 7. Positive and negative controls

- **PC-A (validity gate for H1–H5)**
  - O05 on A2: seeds with CF ≥ 0.90 ≥ **8/10** (pass probability 0.93 at p = 0.9), **and** median CF ≥ 0.95.
  - The median clause is kept only if G2 shows it to be attainable; otherwise it is deleted before the freeze and recorded in DEVIATIONS.
  - Failure → H1–H5 are UNINTERPRETABLE, and the next step is diagnosis.
- **PC-B (control for the HMM pipeline and reproduction of a known effect; one of the gates of H6)**
  - Δ = CF(HQ2-M, 16) − CF(HMM4-C, 19) ≥ 0.10, and BCa 95% CI lower bound > 0. The HMM side has more parameters.
  - Attainability: the exact-fit ceiling is 0.800–0.805, i.e. Δ ≈ 0.18; E2's EM result was 0.619.
  - G5: if CF(HMM4) > 0.88 on any development seed, **stop and diagnose** (this contradicts the known ceiling and indicates a defect). There is no longer a switch to HMM(3).
  - Prediction: pass 0.90.
- **PC-C (reproduction of the E2 failure; an interpretive modifier, not a gate)**
  - Ordered table: mean C0 ≤ 0.30 and ≥ 8/10 seeds < 0.30 → "reproduced"; mean C0 > 0.50 → "not reproduced"; otherwise → "partly reproduced".
  - H1 is adjudicated as usual; PC-C only selects the attribution wording from a fixed table:
    - "reproduced": H1's conclusion may say "repaired the stall reproduced under the E2 protocol";
    - "partly reproduced": as above, with the word "partly" added;
    - "not reproduced": write "on these seeds the E2 protocol itself already reaches a high CF; the need for the repair is not supported".
  - Prediction: reproduced 0.85, partly 0.10, not reproduced 0.05.
- **PC-K (classical training-pipeline control; gate for H6 and STOP, new in this revision)**
  - (a) The 8 cells on A2 shared with E2 (GRU h = 2/4/8/16, LRU n = 2/4/8/16):
    - the average over the 8 cells of (E3 median − E2 mean) ≥ −0.01;
    - and every cell ≥ E2 mean − 0.10.
    - E2 means: GRU 0.210 / 0.288 / 0.741 / 0.809, LRU 0.502 / 0.665 / 0.717 / 0.775 (`runs/e2/SUMMARY.md`).
  - (b) On A1, CPLUS GRU4 median ≥ 0.93 (E2: 0.952), and LRU16 median ≥ 0.858 (E2: 0.888).
  - Attainability: E2 used a constant lr and no aids; CPLUS and CPLUS-M can only help more. G10 first checks (b) on the development seeds.
  - Failure → H6 and STOP are UNINTERPRETABLE.
  - Prediction: pass 0.90.
- **PC-D (fixture sanity; if it does not pass, nothing is run)**
  - CF(exact) = 1 ± 1e-6, CF(marginal) = 0 ± 1e-6;
  - with the true Kraus operators loaded as V·P, error < 2e-3, and gradients finite.
- **PC-O (OOM implementation)**
  - Median CF of OOM(3)-C on A1 ≥ 0.95 (clipped-then-renormalised variant).
  - Attainability: a spectral rank-3 OOM gives 0.981–0.9987.
  - Failure → H7 and D2 are removed.
- **NC (negative control):** the context-shuffled exact predictor has CF ≤ 0.

---

## 8. Statistics and multiplicity

- **Unit of replication:** the confirmatory seed. The arms share data, and the bootstraps share the same resampling of seeds, with random-number seed 20261025.
- **Intervals:**
  - Difference-type rules (H5, H7, PC-B) use **BCa**, 10,000 resamples.
  - If the G8 coverage simulation (per-seed dispersion taken from exploration and E2, paired differences with t5 tails) shows coverage < 0.94 at nominal 95%, these three rules are all switched to 97.5% intervals before the freeze. This is the only pre-registered fallback. [Translator's note: at the development gate the coverage simulation gave 0.882 < 0.94, so this fallback was applied before the freeze; see the development-gate results at the head of this document.]
  - H6 uses the 1st percentile of the percentile bootstrap as its lower bound (see §6).
- **Operating characteristics of the count rules (binomial; labels are determined by counts only):**

| True success rate p | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 |
|---|---|---|---|---|---|---|
| H1 PASS (≥ 9) | 0.011 | 0.046 | 0.149 | 0.376 | 0.736 | 0.914 |
| H1 PARTIAL (7–8) | 0.161 | 0.336 | 0.500 | 0.503 | 0.251 | 0.085 |
| H1 KILL (≤ 6) | 0.828 | 0.618 | 0.350 | 0.121 | 0.013 | 0.001 |
| H3 PARTIAL (6–8) | 0.366 | 0.587 | 0.700 | 0.591 | 0.262 | 0.086 |
| H3 KILL (≤ 5) | 0.623 | 0.367 | 0.150 | 0.033 | 0.002 | 0.000 |
| PC-A count (≥ 8) | 0.055 | 0.167 | 0.383 | 0.678 | 0.930 | 0.988 |

- **Operating characteristics of H2:** P(s ≤ 1 | p = 0.3) = 0.149, P(s ≤ 1 | p = 0.5) = 0.011.
- Unit tests feed every count 0–10 into the summary code, assert that **exactly one label** is returned, and check the values in the table above.
- **Hierarchical gating:**
  1. Validity gates PC-D and PC-A (must pass; no multiplicity correction). PC-C is a modifier.
  2. Family 1 {H1, H2}, count rules.
  3. Family 2: H3a is always adjudicated; H3b is adjudicated only if H1 is PASS.
  4. H4 and H5 are fixed-threshold statements, always reported.
  5. Family 3 {H6, H7}, α = 0.025 per rule. H6 is gated by H1 PASS, PC-B, PC-K and the compute check; H7 is gated by PC-O and H3a not being KILL. A rule whose gate is not passed is removed from the family, and this is reported.
  6. Everything else is descriptive and labelled "exploratory".
- **Job failure** (abnormal exit or LinAlgError, not including NaN; for NaN see §4):
  - Re-run once with the same seed. If it fails again, impute **in the direction unfavourable to the hypothesis**:
    - the tested HQ arm (M) gets CF = 0, unsuccessful;
    - comparison arms (C0, C1, S, GRU, LRU, HMM, OOM) get the maximum CF among the successful seeds of the same arm at the same size.
  - If any arm involved in a rule has failed seeds > 1/10 → that rule is UNINTERPRETABLE.
  - The number of failures and the number of NaNs are reported for each arm.
- **Compute check (one of the gates of H6):** on the confirmatory seeds, the median `cpu_s` of each cell of the decisive ladder ≥ 0.8 × the median `cpu_s` of HQ3-M on A2; otherwise H6 is labelled "not judged (compute mismatch)".
- **Prohibited:** after unblinding, no re-analysis with changed budgets, thresholds or seeds. Every deviation is recorded in DEVIATIONS.md.
- **Order of predictions and summary:** all predictions are written before the freeze, and Brier scores are computed at the end. The summary script first prints the status of every gate; comparisons whose gate is not passed are not output.

---

## 9. Fixture self-checks (stored as `runs/e3/FIXTURE.json`)

**When they are run.** Checks that need trained models (the standard deviation of O05, distinguishability, the planted-truth simulation, the coverage simulation) are **done only on development seeds**, and decisions such as N_test are locked before the freeze. On the confirmatory seeds, after the freeze, only **label-independent** checks are done: CF(exact) = 1, CF(marginal) = 0, mixture consistency, poisoning tests, parameter counts. No design item is changed in the confirmatory phase because of fixture results.

**(i) Denominator artefacts**
- CF(exact) = 1, CF(marginal) = 0, tolerance 1e-6.
- The CF of the mixture prediction λ·exact + (1−λ)·marginal (λ ∈ {0.5, 0.9}) agrees with the direct computation.
- The marginal under the E2 convention differs from the analytical stationary marginal by ≤ 2%.
- The denominators should be: A1 about 0.019, A2 0.0180, A2w 0.0031, A3 about 0.0177.
- The CF of O05 is bootstrapped over the test sequences; its standard deviation must be ≤ 0.01 (A1, A2) or ≤ 0.02 (A2w, A3), checked on development seeds. If this is exceeded, N_test is raised before the freeze to 20k for A2w and to 8k for A3.
- The share of the denominator contributed by k ≤ 3 is reported.
- Fixture test of the H3 adjudication when O05 is broken: with CF(O05) set to −0.05 or 0.3, the seed must be judged unsuccessful.

**(ii) Distinguishability**
- On development seeds, the maximum CF of Markov-1…3 ≤ 0.75.
- A process on which any trivial or linear reference reaches CF ≥ 0.85 may not carry a capability rule with the 0.90 threshold; hence H6 is placed only on A2.
- **Planted-truth simulation (freeze gate, binding):**
  - Method: on planted curves with known truth, run exactly the same median, joint bootstrap and 1st-percentile rule as in §6. Per-seed standard deviations are taken from E2: HQ 0.015, GRU at most 0.08; plus 10% plateau seeds.
  - Criteria: at planted R = 1, the fraction judged "holds" must be ≤ 2.5%; in the censored case planted with "true median CF at the largest rung = 0.92", the fraction mislabelled as holding under censoring must be ≤ 2.5%.
  - If either criterion is not met → H6 is demoted to a descriptive item before the freeze.
  - The detection rate at planted R = 8 is written into the frozen text as found; if it is below 80%, insufficient power is noted.
- **Coverage simulation (freeze gate):** the BCa coverage for paired differences must be ≥ 0.94; otherwise fall back to 97.5% as in §8.

**(iii) Normalisation-convention leakage**
- A check that probabilities sum to 1 is enforced before any metric is computed.
- The model filters from ρ0 = I/d, while the true process starts from the stationary state. For A1 the difference between the two is bounded by 1.03e-7, asserted by a test.
- **Poisoning test:** after the validation, test and exact distributions are replaced by NaN arrays, the fits and selections of MoM, HMM-MoM and OOM-MoM, and the spectral rank r̂, must be bitwise unchanged; validation NaNs may appear only at the step that selects by validation NLL.
- The exact distributions are generated by the data functions and must not be read by training or selection code; this is enforced by the poisoning test. The statement in revision 1 that they are "created only after training" is void.
- On re-evaluation with clipping at 1e-6, CF must change by < 0.005. The rate at which HQ triggers `clamp_min` must be < 1e-4.
- Parameter counts checked against closed forms:
  - A2: HQ 54/96, GRU 63–3843, LRU 59–2563, OOM9 242;
  - A1: HQ2 16, HMM 11/19, OOM3 17, GRU4 130, LRU16 722;
  - A3: HQ4 128.

---

## 10. Fairness measures

1. **HQ has its own controls.**
   - C1 (no warm start, CPU-matched) and S (the literature's default optimiser, likewise CPU-matched);
   - S's τ0 is tuned on the development set, whereas M's recipe is not tuned. This is an asymmetry in S's favour.
2. **The classical families get compute matched in CPU terms**; the argument no longer rests on gradient-step counts:
   - on the decisive ladder, the CPU of each cell ≥ the median CPU of HQ3-M (**including** the MoM fit with 48 restarts), achieved with R_c independent replicas and verified afterwards by the compute check; [Translator's note: deviation D16 in `DEVIATIONS.md` records that this compute check failed in the confirmatory phase (the decisive-ladder cells received 0.66–0.79 times the HQ3-M median CPU); it gated only H6 and, through H6's gates, the STOP rule, and both had already been demoted to descriptive items before the freeze.]
   - GRU and LRU get learning-rate tuning symmetric to S's τ0 grid: G10 selects two values from {1e-3, 3e-3, 1e-2, 3e-2} by development-set validation NLL;
   - LRU has a spectral-λ initialisation, with the rank determined by the data rather than hard-coded to the true value; GRU has no feasible moment initialisation, which is stated as it is, and SENS and D7 check whether the classical curves are limited by optimisation;
   - PC-K checks that the classical pipeline reproduces known levels.
3. **The asymmetry in recipe development is stated as it is.** M's recipe was settled after one day of diagnosis on A2 exploratory seeds; the classical recipes went through no such process. New seeds, a frozen recipe, classical learning-rate tuning and CPU matching mitigate this, but cannot eliminate it.
4. **The HMM gets a symmetric moment initialisation:** the same moment tables and 48 restarts, followed by EM on all 20k sequences, plus 3 randomly initialised EM runs.
5. **The OOM gets a spectral initialisation and a MoM initialisation**, and the candidate set also includes the spectral model before fine-tuning.
6. **Conventions that favour the classical families or the HMM:**
   - P_HQ is taken at the actual size 54, without interpolation;
   - the censored ratio uses the upper bound P_max;
   - in PC-B the HMM has more parameters;
   - failure imputation gives a comparison arm the maximum CF of the same arm at the same size;
   - H6's lower bound uses the 1st percentile;
   - H6's bootstrap resamples the HQ seeds jointly.
7. **Order convention:** taking the true order for HQ and OOM is information given only to the model-structure side. All conclusions state "at the true order"; D4 provides a descriptive check above the true order.
8. **Common conditions:** the same data, the same split, the same loss, the same CF code, the same validation subset, the same normalisation assertion. The oracle-start arm does not enter any capability comparison.
9. **Compute reporting:** for each arm, the median, interquartile range and total are reported, and for each cell the ratio of classical CPU to HQ CPU.

---

## 11. Pre-freeze development gates (development seeds 200–204; 205–209 are reserved only for a single re-run of the gates because of a code fix)

- **G1:** every job type runs end to end on seed 200: no NaN abort, no LinAlgError, all JSON fields present. All old and new tests pass, including the bit-for-bit rerun test, the test that the runs/e2 manifest is unchanged, and the label-exhaustiveness test.
- **G2 (attainability of all PC-A clauses):** on A2, O05 has CF ≥ 0.90 on ≥ 4/5 of seeds 200–204, and median CF ≥ 0.95.
  - Count passes, median fails → before the freeze, PC-A's median clause is deleted and recorded in DEVIATIONS.
  - Count fails → the only permitted change is to extend the fine-tuning of M and C2 to 6,000 steps and re-run once on 205–209; if it still fails, nothing is frozen.
- **G3:** run M, C0, C1 and S on A2; record only, **do not change M**. τ0 is selected by mean development validation NLL; R1 and RS take the smallest value for which development-set CPU ≥ M's development-set median CPU.
- **G4:** the development median CF of O05 on A2w ≥ 0.50; otherwise A2w is removed and H3b is registered as "undecidable (pre-freeze)". In addition, whether 3,000 steps converge on 120k sequences is recorded descriptively (slope of the validation NLL over the last 500 steps).
- **G5 (PC-B calibration):** CF(HMM4-C) ≤ 0.88 on all 3 development seeds; otherwise stop and diagnose, and do not freeze.
- **G6 (PC-O):** OOM(3) has CF ≥ 0.95 on ≥ 2/3 development seeds, and the exact-Hankel fixture passes; otherwise H7 and D2 are removed.
- **G7 (A3, descriptive):** the CPU, CF and O05 of HQ4-M are recorded, for projection only. A3 no longer carries a rule, so no recipe revision is made.
- **G8:** fixtures (i)–(iii) all pass, including the two freeze gates (the planted-truth simulation and the coverage simulation) and the locking of N_test.
- **G9 (compute projection):** measured development-set `cpu_s` multiplied by the number of jobs; development plus confirmatory must be ≤ 28 CPU hours, hard cap 32 (author's decision 1). If exceeded, trim in the following fixed order:
  1. delete the O block (D2);
  2. delete D4;
  3. delete SENS (D7 keeps only the difference between a single replica and multiple replicas);
  4. delete the σ = 0.2 level of C2;
  5. delete A3 (D8);
  6. delete D3;
  7. delete GRU32 and LRU32 (the censoring upper bounds become 1155 and 771, with ratio lower bounds of 21× and 14×).
  - **Termination rule:** if the projection still exceeds 32 hours after all trims, the SC block is not run within E3, and H6 and STOP are pre-registered as "not run (compute)". Thereafter it may be run only after [omitted: organisation or decision-maker reference] separately approves the compute, and then **unchanged, exactly as in the same frozen design**; no redesign is allowed.
  - **Never trimmed:** M, C0, C1, S and C2 (σ = 0.05) on A2, A1-M, PC-B; and, when SC runs, the decisive ladder and PC-K.
- **G10 (classical side):**
  - learning-rate grid {1e-3, 3e-3, 1e-2, 3e-2}: GRU4 and LRU4, seeds 200–201; for each family, two values are selected by mean development validation NLL;
  - a timing probe gives the single-replica CPU at each size (sizes 4 and 8 are interpolated linearly in per-step time between 2 and 32, rounded up); CPU_FLOOR and R_c(F, size) are determined and frozen as constants;
  - PC-K(b) development check: GRU4 ≥ 0.93 and LRU16 ≥ 0.858, each on ≥ 2/3 development seeds; otherwise the pipeline must be diagnosed before freezing;
  - the spectral rank r̂ on A2 is recorded only, not adjusted.

**Freeze and run order:**
1. Compute the SHA-256 of E3-DESIGN.zh.md and of the code manifest, make a git commit [omitted: internal administrative note], add DigiCert and Sectigo timestamps, and write the prediction block.
2. Run the U block with `chain6.sh`. The mechanical summary first prints the gate status and then automatically and idempotently enqueues:
   - the T block: if H1 is PASS or PARTIAL;
   - the SC block: if H1 PASS and PC-B PASS;
   - the O block: if PC-O PASS.
3. After the integrity check passes (code SHA, design SHA, seed ranges, runs/e2 manifest SHA unchanged), the final summary is generated.

---

## 12. Meaning of each outcome (wording constrained by the registered rules)

- **H1 PASS, PC-A passes, H5 established, H2 did not reach the threshold (the most likely case).**
  - At the true order d=3 and N = 20k, a moment warm start computed from training data only, plus fine-tuning with decay, reaches CF 0.90 on at least 9 of 10 new seeds (95% CP lower bound on the success rate 0.61 or 0.74).
  - At the same CPU, random restarts plus decay (C1) are not enough, and this implementation's Stiefel optimiser did not reach the threshold within its τ0 grid.
  - The registered verdicts of E2 remain unchanged. The information E3 adds is that, under recipe M, the low CF of HQ on A2 in E2 is not a capacity problem of the family.
- **H1 and H2 both reach the threshold.** On this process and at this compute, MoM is not necessary; the cheaper route is recommended. The cause is reported only as a hypothesis.
- **H1 PARTIAL.** Transfer is described only, and the SC block is not run; the failing seeds are examined with the moment-loss covariate.
- **H1 KILL and PC-A passes.** The exploration phase was too optimistic; the learnability of HQ on A2 remains an open question.
- **PC-A fails.** H1–H5 are uninterpretable; the next step is diagnosis.
- **PC-C is "not reproduced" or "partly reproduced".** The attribution wording is rewritten according to the fixed table of §7.
- **H5 refuted.** What works is compute plus decay, not the warm start. **H5 undetermined.** No conclusion on attribution.
- **Transfer:**
  - H3a PASS: no harm on A1;
  - H3b PASS: the threshold is reached on both registered instances, A2 and A2w, both at the true order and with N increased under the heuristic rule. This **does not constitute** generalisation within the family.
  - D8 is descriptive only.
- **H4 local.** Compatible with the "optimisation barrier" reading; the plateau mechanism itself is not registered.
- **PC-B or PC-K fails.** The comparison pipeline is still uncalibrated; H6 is not interpreted.
- **H6 holds.** On the A2 process generated by the HQ family itself, at the true order, under the registered rule, the point estimate of the number of parameters GRU and LRU need to reach median CF 0.90 is ≥ 4×54, with lower bound ≥ 2×54. A censored component is written as "not reached within ≤ P_max parameters". This is an internal P(ε) measurement on a realisable process; it is not a separation theorem and does not concern general data. If D7 flags "not robust" or "limited by optimisation", this is stated as well. [Translator's note: H6 and the STOP rule were demoted to descriptive items before the freeze; see the note under H6 in §6.]
- **H6 not established (imprecise, or 2 ≤ R < 4).** With CPU-matched optimisation on both sides, the registered rule did not establish this parameter ratio.
- **STOP triggered.** The sub-direction is closed as in §6.
- **H7 equivalent.** On A1, where an exact low-rank OOM exists, the positivity constraint brings no measurable benefit. "HQ better" always carries "under these training recipes"; if the OOM is worse than its own spectral initialisation, it is relabelled as an OOM optimisation failure.
- **D2.** NR and CF of OOM(9) are descriptive only.

---

## 13. Compute table (CPU hours; low-load estimates, ±30%; G9 re-projects with measured development values)

| Block | Jobs | CPU hours |
|---|---|---|
| DEV (seeds 200–204, including the τ0 grid, the learning-rate grid, timing probes, the PC-K development check, fixtures and simulations) | 84 + fixtures | 3.35 |
| U | 200 | 6.77 |
| T (H1 PASS or PARTIAL) | 33 | 2.15 |
| SC (H1 PASS and PC-B PASS) | 135 | 7.77 |
| O (PC-O PASS) | 10 | 0.69 |
| **Total** | **462** | **about 20.7** |

[Translator's note: these are design-time estimates. At the development gate, G9 projected 34.8 CPU hours without trims and 29.8 hours after all seven registered trims (310 confirmatory jobs), so the termination rule was not triggered; see the development-gate results at the head of this document. Deviation D16 in `DEVIATIONS.md` records about 29.2 CPU hours actually used for development plus confirmatory jobs.]

- **Per-job estimates (CPU-s):**
  - HQ3-M 330, HQ2-M 150, HQ4-M 450, D4 400;
  - C0: A2 115, A1 70;
  - C1 330, S 360, C2 40–70;
  - HMM-C 150, OOM3 150, OOM9 250, REF 40;
  - decisive ladder 330 (= CPU_FLOOR), upper ladder 65–113, SENS 330 (additional replicas).
- **Load risk:** today, at load 10–25, MoM measured about 2× the low-load value (§0, item 6). If the machine is equally busy during the confirmatory phase, CPU as counted by `process_time` may exceed the budget. This is handled by G9's trim order and termination rule.
- If none of the gated blocks runs, the minimum consumption is DEV plus U, about 10.1 hours.
- 3 worker processes, 1 thread per process; wall-clock about 7–10 hours at low load.

---

## 14. Risks

1. **Recipe-selection bias.** M was set after the exploratory seeds had been seen. At p = 0.9, H1's power to return PASS is only 0.74.
2. **The moment objective itself has local minima.** d=4 is descriptive only, with no revision.
3. **The P matrix differs.** This affects only the metric early in fine-tuning.
4. **PC-B's 0.805 is only a bound obtained with EM and the window.** G5 is changed to: stop and diagnose when an anomaly is found.
5. **S is a re-implemented COSM.** A negative result is limited to this implementation.
6. **The heuristic rule for N is not an identifiability argument.**
7. **The MLP readout is not subject to the linear ceiling.**
8. **The values for A3 are exploratory values**, used descriptively only.
9. **NaN risk.** Handled as in §4, with H5 computed in two versions.
10. **Scope:** only the realisable qudit clocks and A1, at the true order; SC only on A2.
11. **Over-interpretation of H6.** Any external text requires separate approval by [omitted: organisation or decision-maker reference] and a claim review, and is not part of E3.
12. **Compute.** With the classical side CPU-matched, SC is the most expensive block; the termination rule may leave H6 unrun.
13. **The median convention is a new choice relative to E2's mean convention.** The reasons are that it is symmetric between HQ and the classical families and robust to a single plateau seed. The E2 comparison values in PC-K are still E2's means, and this is labelled as such.

---

## 15. What is explicitly not claimed

- No claim of "quantum advantage" or "first", and no wording such as "beats". Rule names are only rule names.
- No extrapolation to general data, mismatched processes, real data streams or orders other than the true order.
- H6 concerns only the A2 process generated by HQ itself, at the true order, with the raw count of real parameters as the resource measure, and presupposes H1, PC-B, PC-K and the compute check. It is not a separation theorem.
- No plateau mechanism is registered. The results of H2 do not support a causal claim about the cause of the E2 failure.
- "S did not reach the threshold" does not mean "Stiefel methods cannot"; "MoM works" does not mean "MoM works in general".
- No registered verdict of v1/v1.1, E1 or E2 is changed or reinterpreted. E3 does not request external release. [Translator's note: a later decision publishes the results, together with this English version.]

---

## 16. Code and engineering requirements

- **File scope:** only new files are added, plus an extension of `tests/test_e3.py` and continued use of `src/osw/hq_init.py` (with new helpers only). The code and results of E2 and T1 are not touched.
- **C0 isolation:**
  - C0 calls `run_e2.main` unchanged, with `--out runs/e3/c0` enforced;
  - a test asserts that the file list and SHAs of `runs/e2` are unchanged before and after the E3 queue runs;
  - CPU is recorded with `getrusage` outside the subprocess, and the code-manifest SHA is recorded at the start of each job.
- **Determinism:** the four thread environment variables are set and `torch.set_num_threads(1)` is called; there is a bit-for-bit rerun test.
- **Resumable queue:**
  - atomic writes, by writing a temporary file and then renaming it;
  - keyed by (job key, design SHA, code-manifest SHA); completed jobs are skipped;
  - an attempt counter enforces the "re-run once" rule;
  - runs can be resumed after a restart (a restart occurred on 2026-10-22);
  - gated enqueueing is idempotent.
- **NaN:** a checkpoint is saved every 100 steps; on NaN, revert to the last finite checkpoint and flag it; the number of NaNs is reported per arm.
- **Number of tests:** about 30 new tests; all old tests must still pass.

---

## Review responses

[Translator's note: Review 1 and Review 2 are internal reviews of the design before the freeze, not external peer review.]

| ID | Reviewer and severity | Location | Finding (summary) | Response |
|---|---|---|---|---|
| A-B1 | Review 1, BLOCKER | §6 H1, §8 | H1's labels are not exhaustive and not mutually exclusive; the operating characteristics in §8 ignore the mean condition | Changed to an ordered table based on counts only (KILL ≤ 6 → PASS ≥ 9 → PARTIAL), with the mean and median reported only; §8 uses purely binomial operating characteristics; added unit tests enumerating 0–10 and both kinds of plateau seed |
| A-M1 | Review 1, MAJOR | H3, H4, H5, H6, PC-C | Undefined outcome bands | Every rule gets an exhaustive ordered table: H3 records 6–8 as PARTIAL, H4 and H5 have "undetermined", H6 has "imprecise", PC-C has "partly reproduced"; added grid-enumeration tests |
| A-M2 | Review 1, MAJOR | H6, PC-B, planted-truth simulation | H6 has no classical-side positive control; the planted-truth simulation has no binding force; 5% is inconsistent with α | Added PC-K (non-inferiority test on the A2 cells shared with E2; on A1, GRU4 ≥ 0.93 and LRU16 ≥ 0.858) as a gate for H6 and STOP; the planted-truth simulation becomes a freeze gate with false-call rate ≤ 2.5%, otherwise H6 is demoted to descriptive; PC-B renamed "HMM pipeline control" |
| A-M3 | Review 1, MAJOR | H6 statement | The claim "at least 4 times" goes beyond the inference; P_HQ is held fixed | The statement becomes "point estimate ≥ 4 and lower bound ≥ 2"; HQ seeds are resampled jointly, and resamples with HQ median < 0.90 are recorded as R = 0 |
| A-M4 | Review 1, MAJOR | H6 interpolation, STOP | Left-censoring is undefined; SENS is mixed into H6 and STOP | Under left-censoring, P_F takes the smallest rung (triggering the kill line); CPLUS-M is the only input to H6 and STOP, and SENS is used only for D7 |
| A-M5 | Review 1, MAJOR | C2, H4, PC-A | Noise convention inconsistent with exploration (differs by √2); the relative perturbation is larger on A3 | Registered σ·N(0,1) per real component, written directly into G, consistent with oracle2.py lines 20–21; A3 uses σ = 0.0375 so that the relative perturbations are equal |
| A-M6 | Review 1, MAJOR | PC-A, G2 | The median ≥ 0.95 was not checked for attainability; 9/10 has a 26% failure rate | PC-A count changed to ≥ 8/10 (0.93 at p = 0.9); G2 checks all clauses on 5 development seeds, and the median clause is deleted before the freeze if it is unattainable |
| A-M7 | Review 1, MAJOR | §8 job failure | Imputing CF = 0 favours HQ | The tested HQ arm gets 0; comparison arms get the maximum CF of the same arm at the same size; a rule with failures > 1/10 is UNINTERPRETABLE |
| A-M8 | Review 1, MAJOR | §8 bootstrap | Percentile intervals under-cover at n = 10 | Difference-type rules use BCa, with a coverage-simulation freeze gate and a fallback to 97.5% if it is not met; H6 uses the 1st percentile |
| A-M9 | Review 1, MAJOR | H2 | "≤ 3/10 not repaired" is too wide; the word "reliable" goes beyond the evidence | "Did not reach the threshold" changed to ≤ 1/10, with 2–8 as PARTIAL; all count conclusions report CP bounds; "reliable" deleted |
| A-M10 | Review 1, MAJOR | H7, OOM | The OOM evaluation variant is not fixed; H7's gating depends on H1 | Registered the clipped-then-renormalised variant, with the assertion placed after renormalisation; H3a is always adjudicated, so H7 does not depend on H1 |
| A-M11 | Review 1, MAJOR | H3b, H3c success definition | When O05 fails, success becomes trivial | Seeds with O05 < 0.50 are recorded as unsuccessful, and CF(M) ≥ 0.50 is required; added a fixture test with O05 broken |
| A-M12 | Review 1, MAJOR | §9 second run | The design is changed after the freeze using results on the confirmatory seeds | Only label-independent checks on the confirmatory seeds; checks such as the O05 standard deviation are done only on development seeds, and N_test is locked before the freeze; the assertion about the exact distributions changed to "not read by training and selection code", enforced by the poisoning test |
| A-M13 | Review 1, MAJOR | PC-C in §8 and §7 | Contradictory on whether PC-C is a gate | PC-C becomes an interpretive modifier, removed from "must pass", with the wording chosen from a fixed table |
| A-m1 | Review 1, MINOR | §3, §4 | Wrong pointer to the metric code; wrong statement about batch seeds; some seeds unspecified | Changed to importing queue_e2.metric (lines 62–66) and citing run_e2 lines 126–128; batch seeds stated to be the run seed; a new initialisation-seed namespace covers all recipes |
| A-m2 | Review 1, MINOR | §2 Wilks | Inconsistent units; the statement "identifiability has been matched" is unsupported | Renamed as a heuristic rule; that statement deleted |
| A-m3 | Review 1, MINOR | §5, H4 | 0.20 should be 0.2033, and is a conjecture; the first item of H4 duplicates PC-A | Changed to "≈ 0.20 (0.2033), exploratory, conjecture"; the first item of H4 deleted |
| A-m4 | Review 1, MINOR | H7 interpretation | "HQ better" may come from an OOM optimisation failure | The OOM candidate set includes the spectral model before fine-tuning, selected by validation NLL; if the OOM is worse than its own spectral initialisation, it is relabelled as an optimisation failure |
| B-B1 | Review 2, BLOCKER | H1 | Same as A-B1 | Same as A-B1 |
| B-B2 | Review 2, BLOCKER | H6, CPLUS | The new classical training code has no positive control | Same as A-M2 (PC-K merges the proposals of both reviewers: average non-inferiority on the A2 shared cells, single-cell tolerance 0.10, plus the A1 thresholds); G10 checks first on development seeds |
| B-M1 | Review 2, MAJOR | Fairness, §13 | Counting gradient steps hides the extra compute HQ-M gets; no tuning on the classical side | Decisive ladder CPU-matched, each cell ≥ HQ3-M median CPU (including MoM), with R_c replicas; added a post-hoc compute-check gate; symmetric learning-rate grid on the classical side; C1 and S also CPU-matched; per-cell CPU ratio reported |
| B-M2 | Review 2, MAJOR | Order | All arms use the true order; LRU's rank is hard-coded to 9 | All conclusions state "at the true order"; LRU spectral rank changed to a data-driven singular-value threshold r̂; D4 kept as a descriptive item (limited by compute, not upgraded to H1b); H6 stated explicitly for d=3 |
| B-M3 | Review 2, MAJOR | SENS | How SENS enters H6 is ill-defined | CPLUS-M is the only input; by rule, SENS is placed only at the lower bracketing rung and compared on the same 5 seeds, as a registered robustness check (D7) that does not change labels |
| B-M4 | Review 2, MAJOR | H6 bootstrap | P_HQ is held fixed | Same as A-M3; in addition, all families use the median, making the conventions symmetric |
| B-M5 | Review 2, MAJOR | PC-A, G2 | The median clause is not checked | Same as A-M6 |
| B-M6 | Review 2, MAJOR | PC-B, G5 | Switching to HMM(3) would make the control nearly trivial | Switch deleted; HMM4 > 0.88 → stop and diagnose |
| B-M7 | Review 2, MAJOR | C0 | C0 would write into runs/e2 and contaminate the E2 summary | `--out runs/e3/c0` enforced; test that the runs/e2 manifest SHA is unchanged; CPU recorded with getrusage; code SHA recorded at job start |
| B-M8 | Review 2, MAJOR | Compute | d=4 underestimated, and no termination rule | Measured d=4, O=4 at about 11.9 CPU-s per restart, a ratio of about 1.0–1.15 to d=3, so the scaling of about 4× claimed in the review does not hold; but load inflates process_time by about 2×, so the estimates are raised (M 330, HQ4 450); A3 demoted to a 5-seed descriptive item; G9 gets a seven-step trim order and a termination rule (SC not run; it can only be made up unchanged) |
| B-M9 | Review 2, MAJOR | §12 | The outcome wording goes beyond what the rules support | Rewritten item by item: "reached the threshold (CP lower bound)", "not reached under this implementation", the cause only as a hypothesis, limited to the two instances, "E2 verdicts remain unchanged", censoring written as "not reached within ≤ P_max" with an annotation on the architectural ceiling |
| B-M10 | Review 2, MAJOR | Engineering | Thread settings, a resumable-queue specification and NaN handling are missing | §4 and §16 specify the four thread variables, atomic writes and resumption, attempt counting and idempotent enqueueing; on NaN, revert to the last finite checkpoint, with H5 computed in two versions |
| B-m1 | Review 2, MINOR | Batch seeds | Same as A-m1 | Same as A-m1 |
| B-m2 | Review 2, MINOR | Result bands, H4 | Same as A-M1 and A-m3 | Same as above |
| B-m3 | Review 2, MINOR | A2w, D3 | HQ and LRU train on different numbers of sequences; 3,000 steps are unverified on 120k | Summary tables must not place them side by side; G4 records convergence descriptively |
| B-m4 | Review 2, MINOR | A3, [internal record] (the trigger rule) | A3 is not triggered by a registered failure | Demoted to descriptive item D8, with the reason written into DEVIATIONS [Translator's note: this entry was written only after the freeze; see deviation D15 in `DEVIATIONS.md`.] |
| B-m5 | Review 2, MINOR | H7 convention | The OOM evaluation convention is not fixed | Same as A-M10; "HQ better" always carries "under these training recipes" |
