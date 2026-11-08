# E2 design: at equal parameter counts, do density-matrix (HQMM) sequence models have a measurable benefit over flexible classical models? (English version)

Original: E2-DESIGN.zh.md (Chinese, not published). SHA-256 of the original: b953bbe13e0f896c81754589288cab6b1eafcd538aaf5a85fc87b383dbfd77fa. The original is hash-committed: this hash carries RFC 3161 tokens from DigiCert (2026-10-22 11:51:41 UTC) and Sectigo (2026-10-22 11:51:42 UTC) (`repro/docs/timestamps/2026-10-22-e2-design/`). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

**Status:** Frozen version. Arms A and B of E2 were approved on 2026-10-22 [omitted: organisation or decision-maker reference]. This version is hashed and timestamped before any E2 training.
**When it runs:** After the v1.1 confirmatory phase and E1 have finished.

[Translator's note: The addendum E2.1-ADDENDUM.zh.md (SHA-256 4cec19f1…; English version E2.1-addendum.en.md in this supplement), frozen on 2026-10-22 before any E2 training, changed this: E2 started immediately and ran concurrently with the v1.1 confirmatory phase. The same addendum added two further real data streams, B2 and B3, to arm B (technical report §3.8).]

**Nature:**
- Independent of H1–H5 of v1/v1.1; does not affect their verdicts.
- Arm A has decision rules written in advance; arm B is descriptive only, with one comparison rule written in advance.

## 0. Literature check (2026-10-22; the relevant passages of the original papers were read)

- Srinivasan, Gordon, Boots, "Learning Hidden Quantum Markov Models", AISTATS 2018 (arXiv 1710.09016), §5.1:
  - the probability clock (Zhao & Jaeger 2010) is defined by 2 Kraus operators (equation 16 of the paper) and cannot be represented exactly in any finite-dimensional HMM;
  - the true HQMM has 8 parameters and a description accuracy of 0.163; HMMs with 2, 4 and 8 states reach 0.083, 0.145 and 0.163 respectively, with 8, 24 and 80 parameters.
  - **So "on HQMM-generated data, HQMMs beat HMMs for a given number of parameters" is a known result. This experiment uses it only as a positive control.**
- The Monras 2010 process in §5.2 of the same paper: the state collapses after every observation, the observation sequence is a first-order Markov chain, and a 4-state HMM represents it exactly. It has no long-range structure; this experiment does not use it.
- Adhikary et al., AISTATS 2020 (arXiv 1912.02098): HQMMs are a subclass of observable operator models (OOMs), parameterised on the Stiefel manifold. The HQ model of this experiment uses the same isometric parameterisation.
- An existing local script, [omitted: private file path], deals with "quantum statistical complexity ≤ classical statistical complexity", i.e. an advantage in the number of stored bits; it does not concern prediction accuracy and is complementary to this experiment.
- **What nobody has done yet:** [Translator's note: this reflects the authors' literature search on the date of the design; it is not a priority claim. The technical report's §2 gives the current, checked account of related work.]
  - comparison at equal parameter counts with flexible classical models such as GRU and LRU;
  - measurement per prediction horizon, with the exact predictive distribution as the reference.

## 1. Arm A: synthetic processes most favourable to density matrices (with an exact reference)

**Processes:**
- **A1 probability clock.** 2-dimensional, 2 observation symbols, using equation 16 of the original paper. Relaxation time 2.71 steps.
- **A2 weakly measured three-level (qutrit) clock.** 3-dimensional, 3 observation symbols.
  - Between two measurements, a rotation U = exp(−i·0.6·H), where H = F·diag(0, 1, √2)·F† and F is the 3-point Fourier matrix;
  - the measurement is a Lüders instrument of strength ε = 0.3271, E_y = (1−ε)I/3 + ε|y⟩⟨y|, K_y = √E_y·U;
  - ε was fixed by bisection so that the relaxation time equals 20 steps, aligned with τ = 20 in T1.

**Protocol:**
- Start from the stationary state; context 32 steps; predict k = 1…20 steps; both training and evaluation cover k ∈ [1, 20].
- For each seed, generate 20,000 training, 2,000 validation and 2,000 test sequences; the exact predictive distribution is obtained by HQMM filtering.
- Checks: binned calibration test |z| < 4; with the true Kraus operators loaded into the HQ model, the difference from the exact predictive distribution is < 2×10⁻³ (`tests/test_e2.py`).

**Model families and size ladders (single action):**
- HQ(d): d ∈ {2, 3, 4, 6, 8}. One Kraus operator per observation, all derived from the same isometry.
- GRU(h): h ∈ {2, 4, 8, 16}.
- LRU(n): n ∈ {2, 4, 8, 16}, readout layer width max(4, n).
- HMM(n): n ∈ {2, 4, 8, 16, 32}, EM with 3 restarts.
- Parameter counts depend on the number of observation symbols; for example, at O = 2, HQ 16–256, GRU 54–1,090, LRU 50–722, HMM 5–1,055. Every run records its actual parameter count.
- T1's L/K/C do not take part. They use a shared measurement instrument plus observation-independent dynamics and structurally cannot represent "each observation has its own Kraus operator"; including them would be unfair to density matrices.

[Translator's note: In the implementation the HMM was fitted only on the first 5,000 training sequences or windows, not on the full training set this design specifies (DEVIATIONS.md D14(f); technical report §3.8).]

**Training:**
- Adam, batch 64, 6,000 steps, gradient clipping 5.
- Learning rate {3e-3, 1e-2}; for each (arm, family, size, seed) the better of the two by validation NLL is kept.
- 5 seeds per arm.

**Main metric:** captured fraction CF = 1 − mean_{k≤10} KL(exact‖model) / mean_{k≤10} KL(exact‖marginal), i.e. the fraction of the predictable information that the model obtains.

[Translator's note: As implemented, the marginal is a per-seed empirical approximation of the stationary marginal (technical report §3.8).]

**Comparison:**
- Each family's CF is interpolated linearly in log(parameter count) to fixed budgets; the budgets are {64, 256}.
- Bootstrap over seeds: in each resample all families share the same set of seeds; 10,000 resamples in total.

**Decision rules (fixed in advance):**
- **Main question.** At a given budget, if Δ = CF(HQ) − max(CF(GRU), CF(LRU)) ≥ 0.10 and 95% CI > 0, the verdict is "HQ has a parameter-efficiency advantage at this budget".
  [Translator's note: "95% CI > 0" means that the lower bound of the 95% CI is above 0, as the analysis code implements it.]
- **Stop rule.** If the rule holds for neither A1 nor A2 at either of the two budgets, the verdict is "even on data generated by HQMMs themselves, density matrices have no parameter-efficiency advantage over flexible classical models", and investment in the direction "density matrices bring a predictive advantage" stops.
- **Positive control.** On A1, under the same rule, HQ versus HMM should be judged an advantage; this is the known result of Srinivasan 2018. If the positive control fails, this indicates a problem in our training or comparison pipeline: the conclusion on the main question is not accepted, and the cause is investigated first.

**Expectations stated in advance:**
- HQ versus HMM: an advantage on A1, most probably also on A2.
- HQ versus GRU/LRU: most probably does not hold. LRU's complex diagonal linear recurrence can naturally represent rotations.

## 2. Arm B: a real data stream [omitted: organisation or decision-maker reference] (descriptive only, plus one comparison rule)

**Data:**
- ERA5T weekly mean 2 m temperature, 1.5° grid, 29,040 grid points in total, 134 consecutive weeks (2024-01-15 to 2026-08-03).
- Each week is assigned to category 0–4 against the quintile thresholds of that week's 20-year climatology. This is exactly the quantity forecast in the ECMWF AI Weather Quest.
- The data are read locally only: generated by `scripts/prep_aiwq.py`, SHA-256 `ef583c85…`; the file is not committed to the repository. [Translator's note: "the repository" means the private development repository. The same file is included unchanged in this supplement as `repro/experiments/open-system-world-model/data/aiwq_tas_quintiles.npz`; see `repro/data/README.md`.]

**Split:**
- Context 26 weeks; predict k = 1…8 weeks.
- Training: windows lying entirely within weeks 0–80 (up to 2025-07-28), 20,000 windows.
- Validation: a separately held-out 10% of grid points from the same period, 2,000 windows.
- Test: windows lying entirely within weeks 81–133 (from 2025-08-04, strictly later in time), 5,000 windows.
- Sampling is weighted by cos(latitude).

**Models:**
- The same four families and ladders as in arm A (number of observation symbols O = 5);
- plus three references: uniform climatology (CLIM), training-period empirical frequencies (EMP), first-order empirical transitions (MARKOV1).
- 3 seeds per configuration.

**Main metric:** RPSS over weeks 3–6 [Translator's note: forecast lead weeks, i.e. k = 3–6], i.e. 1 − RPS(model)/RPS(uniform climatology). NLL and ECE are reported as well.

**Comparison rule:** budgets {150, 500}; the rule is the same as in arm A: Δ ≥ 0.10 and CI > 0. The results are descriptive only and are not used for external statements.

[Translator's note: A later decision publishes these results in the technical report (§4.3), where they are reported as descriptive.]

**Known features of the data:**
- Because of the warming trend, the highest quintile accounts for 34.5%, not 20%.
- The category distribution drifts between the training and test periods: the highest category falls from 37% to 31%.
- Both are left to affect the results as they stand; no correction is made.

## 3. Code and tests

- **Code:** `src/osw/tasks_e2.py`, `models_e2.py`, `tasks_aiwq.py`, `run_e2.py`, `queue_e2.py`, `scripts/prep_aiwq.py`.
- **Tests:** `tests/test_e2.py`, 9 in total:
  - calibration of the generators against the exact predictive distributions;
  - relaxation times;
  - the HQ model with the true operators loaded agrees with the exact solution;
  - the weather-data splits are strictly separated;
  - RPS on known values;
  - the decision rule detects a planted effect and does not falsely report a planted null effect;
  - importing the E2 code does not change T1's constants.
- E2 only adds files; it does not change T1's code or results.

## 4. What is not done

- The results of arm A are not extrapolated to data in general.
- The results of arm B are not treated as evidence of [omitted: organisation or decision-maker reference] forecasting skill: the models see only past observations and use no numerical weather prediction.
- No external statement of "better than" of any kind is made.
