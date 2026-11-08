# Pre-registration: a long-horizon calibration comparison of open-system (CPTP) versus unitary-evolution density-matrix world models (English version)

Original: EXPERIMENT-PREREG.zh.md (Chinese, not published). SHA-256 of the original: 37172954eacf4eee0b4e4cf62d031372a7d306f3e6502705f74a5e12c593350c. The original is hash-committed: this hash carries RFC 3161 tokens from DigiCert (2026-10-19 12:09:22 UTC) and Sectigo (2026-10-19 12:09:24 UTC) (`repro/docs/timestamps/2026-10-19-prereg-osw/`). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

**Version:** v1, frozen 2026-10-19 (started directly, with no deadline set [omitted: organisation or decision-maker reference]).
**Freeze procedure:** Once confirmed, take the SHA-256 of this file and of the commit containing the experiment code, obtain RFC 3161 timestamps and OpenTimestamps proofs in the same way as [omitted: private file path], and store them in `docs/timestamps/`. **No confirmatory experiment is run before the freeze.**

[Translator's note: The timestamp folder of this pre-registration (`repro/docs/timestamps/2026-10-19-prereg-osw/`) holds proofs for the SHA-256 of this file only. The code of the confirmatory runs is covered by the code manifest timestamped with the v1.1 addendum (`repro/docs/timestamps/2026-10-21-prereg-osw-v1.1/`).]

**Upstream:** DIRECTION-CARD.md. [Translator's note: This is the planning note, so called below and in the technical report; it is not hash-committed. Its kill tests KT0–KT3 are translated in planning-note-excerpt.en.md.]

[Translator's note: The confirmatory phase was never run under v1. v1 stopped at its development gate, and confirmatory seeds 0–9 were not used under it (DEVIATIONS.md D10). The confirmatory runs in the technical report were made under the v1.1 addendum (EXPERIMENT-PREREG-v1.1.zh.md, SHA-256 d0199a7a…, frozen 2026-10-21), which merges deviations D5–D9; the clauses of v1 that v1.1 does not change remain in force.]

## 1. Hypotheses (primary and secondary distinguished; thresholds fixed in advance)

| # | Hypothesis | Decision rule |
|---|---|---|
| **H1 (primary)** | On synthetic tasks in which information decays, at an equal parameter budget, the CPTP (Lindblad) predictor has a lower mean KL(p_Bayes‖p_model) in the extrapolation window k∈[51,200] than the unitary predictor (UWM-JEPA configuration, d_E=2) | Difference ≥ **0.02 nats**, and the 95% CI over the 10 seeds does not cross 0; tasks: T1-simple, T1-metastable, τ=20 |
| H2 (mechanism: recurrence) | The extrapolation error of the unitary predictor decreases monotonically with the environment dimension d_E∈{1,2,4,8}, and its predictive distribution shows revival peaks over k | Spearman ρ ≤ −0.8, and the revival index (see §4) is higher than that of CPTP |
| H3 (mechanism: beyond ordinary contraction) | On T1-metastable, the learned Lindblad predictor beats the control "unitary rotation + contraction towards a learned fixed state"; on T1-simple the two tie | On metastable, difference ≥ 0.01 nats and the CI does not cross 0; on simple, \|difference\| < 0.01 nats |
| H4 (comparison with classical models) | Compared with the strongest classical baseline (HMM learned by EM, GRU, LRU), the extrapolation KL of CPTP is within ±0.01 nats, or better | Only effect sizes and CIs are reported; no external claim of "better than classical" unless the CI lies entirely on the better side |
| H5 (replication) | Reproduce the UWM-JEPA hidden-velocity task with the public code | Accuracy 0.77 ± 0.05 |
| E1 (exploratory) | Per-horizon log-loss and CRPS curves of each model on ETH/UCY pedestrian trajectories | Descriptive only; no confirmatory conclusion |

[Translator's note: In H1 the difference is KL(unitary) − KL(CPTP); v1.1 writes it as U2 − L.]

**Kill tests** (consistent with the planning note; executed in order; no dates set):
- KT0 novelty.
- KT1 = H5.
- KT2: H1 at τ=20 has a difference < 0.01 nats and a CI that crosses 0.
- KT3: mechanism attribution of H2 and H3.

[Translator's note: Before any run, DEVIATIONS.md D1 redefined KT1 as unit tests of the unitary baseline (unitarity, trace, positivity, symmetry) and H5 as an approximate replication on a reconstructed task, because the public UWM-JEPA repository contains no training code. The H5 approximate replication was never run (technical report §3.6).]

## 2. Tasks

### T1 Synthetic partially observable Markov processes (with an exact Bayes reference)

Hidden states S=8, observation alphabet O=4; the emission matrix B (8×4) is moderately informative and generated with a fixed seed.

- **T1-simple:** P = λ·C + (1−λ)·Π.
  - C is the 8-cycle permutation (pure rotation), Π = 1πᵀ, π is the uniform distribution, λ = e^{−1/τ}, τ∈{5, 20, 80}.
  - Because CΠ = ΠC = Π, Pᵏ = λᵏCᵏ + (1−λᵏ)Π, so the k-step predictive distribution has a closed form: p(o_{t+k}) = b_t·Pᵏ·B.
  - This is exactly the structure "rotation + single-rate relaxation".

  [Translator's note: Simple τ = 5, which no hypothesis uses, was not run (technical report §3.1).]
- **T1-metastable:** two blocks of 4 states each. Fast mixing within a block (τ_fast=3), slow mixing between blocks (τ_slow ∈ {20, 80}). There are two time scales, which a single "contraction towards a fixed state" cannot capture; Pᵏ is computed exactly by matrix power.

  [Translator's note: The emission matrix of this task was changed before any confirmatory run (DEVIATIONS.md D6). A later correction (DEVIATIONS.md, D6 correction, 2026-10-24) found that in the extrapolation window [51, 200] the Bayes predictive of this task relaxes at a single rate (time constant τ_slow), so the two time scales are visible only within the training range.]
- **T1-action:** two actions, each determining the rotation direction of C; used to test action-conditioned generators.
- **Protocol:** start from the stationary distribution, with a context of L=32 observation steps; then a blind rollout: with no new observations, predict the observation distribution for k=1…200 steps.
  - Training set 20,000 sequences, validation set 2,000, test set 2,000.

    [Translator's note: In the implementation the HMM was fitted only on the first 5,000 training sequences (DEVIATIONS.md D14(f)).]
  - **Training uses only k∈[1,50]; evaluation covers k∈[1,200].** Beyond 50 is the extrapolation window; this is the key design feature of this experiment.

### T2 UWM-JEPA hidden-velocity task

Use the original authors' public code (https://github.com/santoshkumarradha/uwm-jepa, Apache-2.0, keeping the NOTICE and attribution), with the task and hyperparameters as in the original paper; in addition, a version with the predictor replaced by a CPTP predictor serves as a comparison.

### E1 Real data stream (exploratory)

The ETH/UCY pedestrian data already in the repository (`data/ethucy`). The future direction of travel is discretised into 8 sectors; leave-one-out cross-validation by scene (LOSO) reuses the split of [internal record] (an earlier ETH/UCY pedestrian-trajectory experiment in the repository).

[Translator's note: The details of E1 were fixed later in a separately frozen design (E1-DESIGN.zh.md, SHA-256 f06879ca…; English version E1-design.en.md), which exists no later than its RFC 3161 tokens (2026-10-21 15:51 UTC); the repository history records the first E1 code after that time.]

## 3. Models (all models use the same context and the same training loss)

| Code | Structure | Notes |
|---|---|---|
| U(d_E) | Joint state ρ_J ∈ ℂ^{8d_E×8d_E}; each step U(a) = exp(−iH(a)); in the context phase, conditional update with a measurement instrument on the system, ρ ← (M_o⊗I)ρ(M_o⊗I)†/p(o); readout p(o) = Tr[(E_o⊗I)ρ] | d_E∈{1,2,4,8}; d_E=2 is the UWM-JEPA configuration |
| L | System state ρ ∈ ℂ^{8×8}; the GKSL generator consists of H(a) plus J jump operators L_j(a); per-step channel exp(Δt·𝓛), computed as a 64×64 matrix exponential; measurement instrument as above | Reuses the conventions of [omitted: private file path], with regression tests against the closed-form qubit solution |
| K | Discrete Kraus channel (HQMM-style) with r Kraus operators; a Stiefel parameterisation guarantees CPTP | Established quantum-inspired approach |
| C-ctrl | Φ(ρ) = e^{−γ}·Uρ U† + (1−e^{−γ})·σ, where σ is a learned fixed state | "Ordinary contraction" control (responds to blind reader 3) [Translator's note: one of three independent readers who were given only the design text of the planning note; see planning-note-excerpt.en.md.] |
| HMM | 8 states, learned by EM | The true model class; should be close to Bayes-optimal on T1; used as the yardstick of "how good" |
| GRU / LRU | Classical latent-vector models that output a softmax predictive distribution | Classical baselines |
| Bayes | Exact forward algorithm plus Pᵏ | T1 only; not ranked, reference only |

[Translator's note: In the implementation only H(a) depends on the action; the jump operators L_j do not (DEVIATIONS.md D12, disclosed 2026-10-22). This matters only for T1-action.]

**Equal budget:** within each comparison, the numbers of trainable parameters differ by no more than ±10%. Where this cannot be achieved, both "native size" and "parameter-matched" results are reported. The computation per forward pass (FLOPs) is also reported.

## 4. Training, metrics and statistics

- **Training:**
  - Optimiser Adam; learning rate only from {1e-3, 3e-3}, selected on the **validation set**; 2,000 steps, batch 64.
  - The loss is the multi-horizon negative log-likelihood (NLL), with k sampled uniformly from 1–50; identical for all models.
  - Everything runs on CPU with PyTorch (which supports complex matrix_exp) and Python 3.11; dependency versions are pinned.
- **Per-horizon metrics:**
  - KL(p_Bayes‖p_model), T1 only;
  - NLL, Brier score, ECE (10 bins);
  - E1 additionally reports CRPS.
- **Windows:** within the training range [1,50]; extrapolation window [51,200].
- **Revival index (H2):** let D(k) = TV(p_model,k, p_model,∞), where p_model,∞ is the average over k∈[400,600]. Revival index = number of local maxima of D(k) on [51,200] × mean peak height.

  [Translator's note: DEVIATIONS.md D4 (2026-10-19, before any confirmatory run) specifies the computation: for each sequence, the number of strict local maxima with D > 1e-6 times their mean height, then averaged over 500 test contexts.]
- **Statistics:**
  - 10 seeds; each seed generates its data independently and is initialised independently.
  - For each seed, first take the window average, then form paired differences by seed; 95% CIs are given both as bootstrap intervals (10,000 resamples over seeds) and as t intervals.
  - H1–H4 are treated as one family, with Holm correction.
  - Effect sizes are always reported (unit: nats).
- **Report everything:** positive, negative and null results all go into the technical report, without selection.

## 4b. Development seeds and confirmatory seeds

- **Development seeds 100–109:** used only for debugging, unit tests and learning-rate selection (on the validation set); their results do not enter the confirmatory conclusions.
- **Confirmatory seeds 0–9:** after the code passes all unit tests and the development runs are stable, each configuration is run only once. The commands, code commit IDs and output hashes of the confirmatory runs are all recorded in `experiments/open-system-world-model/runs/`.

## 5. Deviations and transparency

- Any change after the freeze is recorded in `DEVIATIONS.md`, with its reason and date; analyses obtained after a change are always labelled "exploratory".
- External wording is first checked against [omitted: internal project name]: no "first", no generalised statement of "better than the baselines", no "quantum advantage"; "better than" only when the CI lies entirely on one side, and restricted to the specific task.

## 6. Compute and time

- At d=8 the Liouvillian is 64×64, and at d_E=8 the unitary is 64×64; a single run is expected to take under 5 minutes.
- All combinations: about 3 task families × 8 models × 10 seeds × 2 learning rates ≈ 480 runs, about 1–2 days of CPU with multi-core parallelism.
- External cost $0.

## 7. Output ladder (corresponding to the tests of the planning note)

1. **After KT3:** KT0–KT3 have conclusions → publish a technical report on Zenodo with a DOI; publish whether the results are positive or negative. The benchmark is a citable asset.
2. **After the technical report:** if H1 holds and the mechanism can be attributed (H2/H3) → submit to TMLR.
3. **Conditional:** a quantitative lower-bound theorem is proved + E1 shows a robust gain → redo the three blind reads for the new design, then decide whether to aim for a top-tier conference.
