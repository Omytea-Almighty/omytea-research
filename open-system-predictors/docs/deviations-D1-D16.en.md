# Deviation record D1–D16 (English version)

Original: DEVIATIONS.md (Chinese, not published). SHA-256 of the original: b322cc3815e3214cb55befee6b57e7e9fe93781c8ede9b348f524977730e4fa3. The original is not hash-committed and carries no RFC 3161 timestamp. Its history is kept in the private repository, and no line has ever been deleted or changed; entries have only been added. The hash above is of the file as it stood on 2026-10-30. The latest repository commit of the file (2026-10-24) ends at D14(f), so D15, D16 and their dated additions are not yet in the repository history. A later section of the file, on an unpublished follow-up experiment, is not part of this publication. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

# Record of deviations from the frozen pre-registration

Pre-registration v1: SHA-256 `37172954…`, frozen 2026-10-19T12:09:22Z. Every deviation is written down **before** the relevant experimental runs, with its reason stated.

[Translator's note: this sentence dates from the first entries. D12–D16 and the dated corrections were written after the relevant freeze, several of them after the runs concerned; most headings state when the entry was written (the D2, D3 and D10-supplement headings carry no date).]

## D1 — KT1/H5 changed to "implementation to the paper's specification + approximate replication" (2026-10-19, before any run)

**Facts:**
- The UWM-JEPA repository (commit `1ef5735`, Apache-2.0) contains only aggregated result JSON files and plotting scripts, **no training code**; yet the paper's main text says "All code, data, and figure-generation scripts are available".
- The environment of the hidden-velocity task ("CartPole-style spring benchmark") has no published parameters: the dynamics constants, the noise and the way actions act are not given.

**Changes:**
1. The unitary baseline U(d_E) is implemented strictly to the specification in the paper's §Belief-State JEPA:
   - joint dimension 8×d_E, initial state the maximally mixed state;
   - observation update: sandwich with the system measurement operator K(o)⊗I, then normalise the trace;
   - prediction: unitary conjugation, H(a) = H₀ + a·H₁.
   - The correspondence between the paper's interface and this implementation is recorded in comments in `src/`.
2. The hidden-velocity task becomes an **approximate replication**: following the paper's description, a "partially observable inverted-pendulum-style oscillator with a spring" is rebuilt (observation stride 3, K=5); the environment parameters are written out in `src/tasks_spring.py`.
   - Reaching 0.77 ± 0.05: recorded as "approximate replication succeeded".
   - Not reaching it: **REDESIGN is no longer triggered**, because it would be impossible to tell whether our environment differs or the original conclusion itself does not hold; the result is only reported as it is.

   [Translator's note: the H5 approximate replication was never implemented or run, and `src/tasks_spring.py` was never written; the technical report (§3.6) reports no H5 result.]
3. The role of KT1 becomes: confirming that the unitary baseline is implemented correctly. The criterion is unit tests (unitarity, trace, positivity, symmetry), not reproducing some accuracy.
4. Optional: ask the original authors for their training code. This would be external contact [omitted: organisation or decision-maker reference]; not done.

**Why this does not affect the main conclusion:** the main hypotheses H1–H3 are all tested on the T1 synthetic tasks, which have an exact Bayes reference and do not depend on the original authors' environment. In addition, the limitations part of the UWM-JEPA paper states "the present results do not extend to hidden Markov chains, ring-walker environments", and T1 is exactly a hidden Markov chain on a ring, i.e. precisely the region the original authors themselves marked as not covered.

## D2 — Training objective unified as multi-horizon NLL (clarification, not a deviation)

Pre-registration §4 specifies the same multi-horizon NLL for all models. U(d_E) is therefore trained with NLL, not with the original paper's JEPA latent-matching loss. What is compared is the **predictor dynamics**, not the training framework. The report will state "UWM-style predictor, trained with NLL".

[Translator's note: D14(b) later records that the HMM was fitted by EM on the joint likelihood of whole observation sequences, so "the same loss for all models" holds only for the neural-network and density-matrix models.]

## D3 — The density-matrix models use the same quantum instrument for readout and update (implementation detail)

- Measurement operators {M_o}: obtained from A_o by normalisation with S^{−1/2}, so that Σ M_o†M_o = I.
- Readout: p(o) = Tr(M_o ρ M_o†).
- Update: ρ ← M_o ρ M_o† / p(o).

The four density-matrix models U, L, K and C-ctrl all use this same parameterisation; they differ only in the dynamics between two observations: unitary / GKSL / Kraus / contraction.

## D4 — Concrete algorithm for the revival index (clarification, before any confirmatory run, 2026-10-19)

Pre-registration §4 only says "number of local maxima of D(k) on [51,200] × mean peak height", without saying whether this is computed per sequence or on the mean curve. It is now fixed as follows:
- first compute it for **each sequence** separately: the number of strict local maxima (with D > 1e-6), multiplied by the mean height of those peaks;
- then average over the 500 test contexts.
- p_model,∞ is taken as the mean over k∈[400,600], so long-horizon rollouts are computed up to k=600. This uses a separately drawn set of 500 contexts, with random seed `10000·seed + task_key + 777`.
- Two auxiliary quantities are reported as well: the mean number of peaks, and the mean of D(k) over [51,200].

If runs already started in the development phase do not yet have this field when they are produced, this does not affect learning-rate selection (which uses validation NLL only).

## D5 — Training settings (2026-10-21, written after the first development round and before the confirmatory runs; the concrete values to be fixed after the seed-101 comparison)

**Facts** (development seed 100, `runs/dev/`, `GATE.md`):
- With 2,000 steps, learning rates {1e-3, 3e-3} and the instrument initialised "near identity", the KL within the training range on T1-simple:
  - the density-matrix models for which an exact solution exists (Proposition 2) stall at about 0.03: L 0.0298, K 0.0290, C 0.0305;
  - GRU is 0.0025, HMM 0.0000.
- U1 "learned nothing" at both learning rates; but a hand-constructed mixture of a rotation and the marginal distribution reaches NLL 1.341 < ln 4 = 1.386, which shows that this is an optimisation problem, not a capacity limit.

**Why change:** comparing U and L with under-trained models measures optimisation speed, not the difference between dynamics families that the pre-registration sets out to measure.

**Change:** on the new development seed 101, four settings are compared:
1. the current setting;
2. learning rate 1e-2;
3. random instrument initialisation;
4. 6,000 steps.

The setting that makes L and K converge (KL within the training range ≤ 0.01) is selected and **used uniformly for all models**. The results and the final choice are written at the end of this section.

**Unchanged:** H1–H5, the thresholds, the extrapolation windows and the statistical methods are all left unchanged.

In addition, a semantics-preserving performance patch: the operators are applied separately for each action and the result is then selected, instead of copying the operators for every sample. All 28 tests pass; time per step U2 251→137 ms, L 410→154 ms, U8 2078→1298 ms.

**seed-101 results** (simple τ=20, `runs/probe/`; KL is the mean over the training range [1,50]; validation NLL in brackets):

| Setting | L | K | U2 | U1 |
|---|---|---|---|---|
| ① 3e-3 × 2,000 | 0.0296 (1.3439) | 0.0203 (1.3358) | 0.0452 (1.3617) | 0.0693 (1.3869, learned nothing) |
| ② 1e-2 × 2,000 | 0.0123 (1.3273) | **0.0036** (1.3193) | 0.0514 (1.3706) | 0.0704 (1.3880, learned nothing) |
| ③ 3e-3 × 2,000, random instrument initialisation | 0.0286 (1.3426) | 0.0280 (1.3432) | 0.0428 (1.3580) | 0.0691 (1.3865, learned nothing) |
| ④ 3e-3 × 6,000 | 0.0294 (1.3443) | **0.0031** (1.3185) | 0.0423 (1.3570) | 0.0402 (1.3549) |
| ⑤ 1e-2 × 6,000 (added) | **0.0034** (1.3193) | **0.0019** (1.3179) | 0.0367 (1.3523) | 0.0206 (1.3360) |

**Interpretation:**
- At learning rate 3e-3, L stalls on a plateau at KL≈0.03: going to 6,000 steps or changing the initialisation does not help; only learning rate 1e-2 gets it out.
- U1 learns something only at 6,000 steps.
- U2 is better at learning rate 3e-3.

No single learning rate serves all models at once, but the pre-registration already specifies "select the learning rate for each model on the validation set".

**Final choice** (written to `runs/TRAIN_CONFIG.json`, used uniformly for all models):
- **6,000** steps;
- learning-rate candidates **{3e-3, 1e-2}**, selected for each (task, τ, model) by validation NLL on development seed 102;
- instrument initialised near identity; batch 64; Adam.

Under this setting, L and K both have KL within the training range ≤ 0.01 on seed 101; ROUND 2 GOAL (a) is met.
- This judgement rests on the development seeds only;
- the confirmatory phase does not re-check this engineering goal.

**The added setting ⑤ (1e-2 × 6,000)** is a second attempt within the ROUND 2 contract. None of the original four settings made L converge:

```
GOAL=L,K in-range KL<=0.01 on seed 101 | VARIANT=L in-range KL (0.0123 best of four) | INVARIANT=H1-H5, thresholds, windows, confirm seeds untouched
VERIFIER=seed-101 probe JSON | FUEL=4 jobs (~10 CPU-min each), rounds 2/3 | COMBINE=pick setting by L,K convergence; per-model lr by val NLL
ESCALATE=if L still >0.01, stop [omitted: organisation or decision-maker reference] (no threshold change)
```

[Translator's note: the block is the author's pre-declared contract for this second attempt, in internal shorthand: GOAL = target; VARIANT = the quantity tracked (L's best in-range KL so far); INVARIANT = what may not change; VERIFIER = the evidence used; FUEL = the compute budget and round count; COMBINE = the selection rule; ESCALATE = the stop rule. "ROUND 2 GOAL (a)" above is the target in the GOAL field: L and K reach in-range KL ≤ 0.01 on seed 101.]

## D6 — Emission matrix of the metastable task (2026-10-21, before the confirmatory runs)

**Facts:**
- The original emission matrix makes states s and s+4 emit the same symbol, and the two blocks are exactly {0–3} and {4–7}, so block identity is completely unobservable.
- As a result, the information of the Bayes prediction relative to the marginal distribution is only 0.004 nats (within the training range) and 0 (extrapolation window); the task is degenerate and H3 cannot be tested.

**Change:** block A emits symbol s mod 2 and block B emits 2 + (s mod 2), with probability 0.7, and each of the other symbols with probability 0.1 (`tasks.emission_metastable`). The emission matrices of T1-simple and T1-action are unchanged.

**Signal after the change (KL(Bayes‖marginal)):**

| τ_slow | Training range [1,50] | Extrapolation window [51,200] |
|---|---|---|
| 20 | 0.0287 | 0.00006 |
| 80 | 0.0977 | 0.0121 |

**Consequent arrangements:**
- **H3 is tested at τ_slow=80.** The pre-registered H3 does not specify τ, while §2 lists τ_slow ∈ {20, 80}; only at 80 are both time scales visible in the extrapolation window.

  [Translator's note: the D6 correction of 2026-10-24, later in this record, states that this last sentence is wrong: in the extrapolation window [51, 200] the Bayes predictive of this task relaxes at a single rate (time constant τ_slow), for τ_slow = 20 and 80 alike. The registered verdicts are unchanged.]
- **H1 is still tested at τ=20 as pre-registered**, unchanged.
- **T1-simple τ=80 is reported as a secondary result**, clearly labelled; it is in the task list of §2, and its extrapolation-window signal is 0.033.

## D7 — Scope of the gate's "learned nothing" criterion (2026-10-21, before the confirmatory runs)

This criterion originally applied to all models. It now applies only to the models for which a learnable solution is known to exist and which should learn it in the development phase: L, K, C, GRU, LRU.

For the U family:
- **U2 must learn something:** at the selected learning rate, validation NLL < ln 4 − 0.003. U2 is the original UWM-JEPA configuration and the comparator of H1; comparing an under-trained unitary model with L would be an unfair comparison.
- **U1, U4, U8 are only required to give finite results:** whether or not they can learn is itself the phenomenon H2 looks at.

This is only an engineering gate, not pre-registered content; it does not loosen the adjudication of any hypothesis.

### D6 supplement (2026-10-21): observability check of the changed task

The criterion of finding D02 of the read-only diagnosis [omitted: private file path] is adopted: before any mechanism test, first confirm that the slow mode is visible in the observations.

[Translator's note: D01–D06 are the diagnosis's own finding labels, not deviations D1–D16. A summary of the diagnosis is included in this English supplement package (docs/development-gate-diagnosis-summary.en.md).]

- **Original emission matrix:** the slow block-difference mode v satisfies vB = 0, so the whole observation process can be lumped into a 4-state single-rate chain (diagnosis finding D02).
- **New emission matrix (`emission_metastable`):** a read-only check of |vB|, the product of the left eigenvectors of P with the emissions:
  - τ_slow = 20: the visible non-stationary time constants are **2.61 and 19.99**;
  - τ_slow = 80: the visible non-stationary time constants are **2.89 and 80.14**;
  - the within-block rotation modes ±i·0.68 are not visible; this affects only the within-block phase, not the H3 premise of "two identifiable decay rates".

  [Translator's note: per the D6 correction of 2026-10-24, the fast mode (2.61 or 2.89) has decayed by k = 51, so the two decay rates are visible only within the training range, not in the extrapolation window.]
- Candidate A of the diagnosis (B + 0.05·v·cᵀ) and candidate B [omitted: internal administrative note] were both **not adopted**. This experiment adopts a third option (block identity distinguished by pairs of symbols), for the reasons and with the values given above.

## D8 — Corrections to the confirmatory analysis (2026-10-21, before any confirmatory run)

Diagnosis finding D06 is adopted. `src/osw/analyze.py` is modified, and a new test file `tests/test_analysis.py` (6 tests) is added:

1. **Seed completeness:** every test must have all 10 confirmatory seeds; otherwise it is judged INCOMPLETE and cannot PASS. The original code could already produce intervals and a PASS with only 2 seeds.
2. **PASS criterion:** all four of the following must hold at once:
   - the pre-registered effect-size threshold;
   - the bootstrap 95% CI does not cross 0;
   - the t-interval 95% CI does not cross 0;
   - **Holm-corrected p < 0.05**. The original code's PASS did not look at Holm.
3. **Holm family:** H1 (two tasks), H2, H3, H4 (three tasks), 7 tests in total. The pre-registration says "H1–H4 are treated as one family".
   - For conjunctive hypotheses (H2: Spearman plus revival; H3: metastable plus the tie on simple), the maximum of the component p-values is taken, i.e. an intersection–union test.
   - Including H4 can only make H1–H3 harder to pass.

   [Translator's note: D14(a) later records that the frozen `analyze.py` did not apply this conjunction rule to H2's rank component (it used a fixed threshold, with no p-value); see also the D13 correction. Under the registered conjunction, H2 is not met.]
4. **H3:**
   - tested on metastable τ_slow = 80 (D6);
   - the comparison is with the parameter-matched L1 (D9);
   - the "tie" on simple becomes an equivalence test (TOST, margin ±0.01 nats), stricter than the pre-registered point-estimate rule;
   - the same comparison with the original-size L is a secondary result and does not enter the Holm family.
5. **H4:**
   - the strongest classical baseline is now selected by **validation NLL on the development seeds**, which removes the bias of selecting the baseline with confirmatory data;
   - "comparable or better" becomes a non-inferiority test (margin 0.01 nats). If development data are missing, the script stops with an error and does not fall back to confirmatory data.

   [Translator's note: D14(c) later records that the HMM had no recorded validation NLL, so this selection was in effect made between GRU and LRU only.]

**Under pre-registration §5:** analyses obtained after these changes are all labelled "exploratory" relative to v1; in addition, they are frozen by the v1.1 pre-registration addendum before the confirmatory runs (D10).

## D9 — Parameter-budget matching (2026-10-21, before any confirmatory run)

Pre-registration §3 requires that the parameter counts in each compared group differ by no more than ±10%. Diagnosis finding D06 [Translator's note: D06 is a finding label of the read-only development-gate diagnosis, which is summarised in this supplement; it is not deviation D6.] pointed out several places where this does not hold: C 769 against L 1024; GRU 1220 and LRU 1188 against L 1024. They are now changed as follows (one action / two actions):

| Model | Before | After | Difference from the comparator |
|---|---|---|---|
| GRU | hidden size 16: 1220 / 1268 | hidden size 15: 1099 / 1144 | vs L: +7.3% / −0.7% |
| LRU | state size 24: 1188 / 1236 | state size 20: 1004 / 1044 | vs L: −2.0% / −9.4% |
| L1 (new) | — | 1 jump operator: 768 / 896 | vs C (769 / 897): −0.1% |

- **H1**'s U2 against L (one action, 1024 against 1024) was already matched; unchanged.
- **H3**'s main comparison becomes C against L1 (parameter-matched); the original-size L is reported as a secondary result.
- **Comparisons that are not matched but enter no hypothesis are reported as they are:**
  - in the action task, U2 and K have 1536, against 1152 for L;
  - U4 and U8 are H2's deliberate sweep to larger dimensions.
- Added the test `test_parameter_budgets_match_within_ten_percent`; all 34 tests pass.

## D10 — v1 stops at the development gate; switch to the v1.1 supplementary pre-registration (2026-10-21)

- **Status of v1:** development gate FAIL (`runs/dev/GATE.md`). Pre-registration §4b stipulates that the confirmatory seeds are run only after the development runs are stable; therefore **v1 does not enter the confirmatory phase**, and the confirmatory seeds 0–9 were never used. The technical report will state this as it is. [Translator's note: seeds 0–9 were later used as the confirmatory seeds under v1.1 (technical report §4.1).]
- **v1.1:** D5–D9 are combined and written up as the supplementary pre-registration `EXPERIMENT-PREREG-v1.1.zh.md`. After the round-2 development gate passes and before any confirmatory run, it is hashed and given an RFC 3161 timestamp.
- **Labelling:**
  - relative to v1: labelled "exploratory", in compliance with v1 §5;
  - relative to v1.1: confirmatory;
  - the report gives the two labels side by side and does not choose between them.

## D11 — Run-time diagnostic records, and a naming defect in the round-2 gate (2026-10-21, before the confirmatory runs)

1. **Run-time records:** `run.py` adds four records:
   - the initial validation NLL;
   - the gradient norm (before clipping) every 200 steps;
   - parameter checkpoints;
   - the SHA-256 of the v1.1 file.

   **Why:** this adopts the comment in §6 of the report outline [Translator's note: an internal outline drafted for this technical report; not published.]. Without this evidence, under-training, a local optimum and insufficient representational capacity cannot be told apart.

   **Training is not changed:**
   - a unit test compares the two cases, with and without the records: parameters and losses are bit-identical;
   - the dev2 job for simple τ=20, GRU, seed 102 was re-run with the new code: all metrics are bit-identical to the original record.
2. **Naming defect [internal record]:**
   - Symptom: the round-2 development gate first returned FAIL, reporting 10 HMM/BAYES files as "missing";
   - Cause: `queue.out_name` adds no step-count tag for HMM/BAYES, whereas `run.py` adds it for all models;
   - Handling: all 80 result files already existed at that time; after the naming was fixed, the gate was re-run and returned PASS (`runs/chain2.log`); no result was changed;
   - Regression test: a new end-to-end test that actually runs `run.main` and checks the file names. It fails on the old code and passes on the new code.
3. **Logging defect [internal record]:** the exit code recorded by `chain2.sh` was in fact that of `date`; fixed.

## D10 supplement — v1.1 freeze

After the round-2 development gate PASS, the text of v1.1 was finalised and timestamped, and only then was the confirmatory phase started. For the timestamps see `docs/timestamps/2026-10-21-prereg-osw-v1.1/`. [Translator's note: in this supplement, `repro/docs/timestamps/2026-10-21-prereg-osw-v1.1/`.]

## D12 — The jump operators of L do not vary with the action (disclosed 2026-10-22; found after the v1.1 freeze, while the confirmatory phase was running)

- **Fact:** pre-registration v1 §3 says "the GKSL generator consists of H(a) plus J jump operators L_j(a)". In the implementation `LindbladWM`, only the Hamiltonian varies with the action (H(a) = H₀ + aH₁); the jump operators L_j do not vary with the action. This was found while drafting the methods section of the technical report and checking the code line by line.
- **Scope of the impact:**
  - It affects only T1-action (two actions). T1-simple and T1-metastable have only one action, and there the two formulations are fully equivalent.
  - Hence H1, H2 and H3 are unaffected; only the action part of H4 (report-only) and secondary descriptions are affected.
- **Handling:**
  - The code is not changed. v1.1 and the code manifest are frozen, and the confirmatory runs are under way.
  - In the technical report, action results that involve L are annotated with this difference: the dissipative part of L cannot change with the action, which constrains L rather than helping it.
  - No re-run, and no threshold is adjusted.

## D13 — Floating-point error in the H2 adjudication (2026-10-22, found after the confirmatory analysis was complete)

- **Fact:** the frozen `analyze.py` computes the Spearman correlation of the 4 points in floating point, obtains −0.7999999999999999, compares it with −0.8, and judges H2 NOT MET. Computed exactly from the ranks: Σd² = 18, ρ = 1 − 6·18/60 = −4/5, which exactly satisfies the pre-registered rule "ρ ≤ −0.8".
- **Handling:**
  - the output of the frozen code is kept as it is (`runs/confirm/RESULTS.md`);
  - the adjudication made according to the letter of the rule is written in `runs/confirm/RESULTS-ADJUDICATION.md`; [Translator's note: that original is not published; its English version is `docs/T1-results-adjudication.en.md`.]
  - the report gives the two side by side, rather than only the one favourable to the conclusion.
  - Also reported: this adjudication falls exactly on the boundary, and in substance it does not support a monotone relation (see that file).
- **Code fix:** `analyze.py` now uses the exact rational rank formula, and a regression test is added (exactly at the boundary is judged as satisfied). The fix applies only to future analyses and does not rewrite the frozen confirmatory output.

## D13 correction (2026-10-23, after the round-2 two-model review)

The statement in D13 that "by the letter of the rule, H2 is PASS (exactly at the boundary)" is **wrong**. Two review legs from different model families, which did not communicate with each other, each pointed this out independently [omitted: private file path]. [Translator's note: the review legs in this file are the author's internal correctness reviews, each run with an AI model from a different model family; they are not external peer review. In the Source column of D14 below, the two legs are labelled leg A and leg B in place of the omitted names. The internal review records are not part of this supplement.]

- v1.1 §6 stipulates that a conjunctive hypothesis takes the maximum of its components' p; item 3 of D8 explicitly writes H2 as two components, "Spearman plus revival".
- With 4 points, the exact one-sided permutation test of the rank component gives p = 1/6; the smallest p that any 4-point rank test can give is 1/24.
- Hence H2's Holm-adjusted p after the conjunction is 1/3, and **H2 does not hold**. The conclusion is the same whether ρ is computed in floating point or in exact arithmetic.
- D13 corrected only the threshold comparison; it did not carry out the conjunction test that the rule requires. Exact arithmetic shows only that "as an effect size, the threshold holds exactly at the boundary"; it cannot decide the verdict on H2.

## D14 — Implementation departures found in review (2026-10-23, after the confirmatory results were read out; all are disclosures)

| No. | Departure | Source |
|---|---|---|
| (a) | The frozen `analyze.py` treats H2's rank component as a fixed threshold and gives it no p-value, so H2 in the Holm family carries only the p of the revival component. This departs from the conjunction rule specified in v1.1 §6 and D8. The report's Table 1 gives both sets of Holm values, and H2 is adjudicated as not holding according to the letter of the rule. | Round-2 review, both legs |
| (b) | The HMM is fitted by EM to the joint likelihood of whole observation sequences, using 5,000 sequences and 3 restarts, rather than by the blind multi-horizon NLL shared by the neural-network and density-matrix models. The "same loss for all models" stated in D2 holds only for the latter. | Round-2 review, leg A [omitted: internal tool name] |
| (c) | The HMM recorded no validation NLL, so H4's baseline selection was in effect made only between GRU and LRU. The report gives a sensitivity analysis that includes the HMM among the candidates (post hoc). | Round-1 review, both legs |

## D14 supplement (2026-10-23, after the round-3 two-model review)

| No. | Departure or correction | Source |
|---|---|---|
| (d) | **Inconsistent training window in E1.** The HMM (EM) is fitted on whole training sequences: the context plus all 12 future steps, so it also sees steps 7–12 of the **training** sequences (the test sequences are not seen). The two references MARG and MARKOV1 are also estimated on whole training sequences. The frozen design's rule "training uses only steps 1–6" holds only for the gradient-trained models. Post hoc sensitivity analysis: after the HMM is refitted on the context plus only the first 6 future steps, its extrapolation NLL changes from 1.157 to 1.200, on average 0.020 worse than L, with large differences between scenes. | Round-3 review, leg A [omitted: internal tool name] |
| (e) | **Post hoc correction of `analyze.py`.** It now performs H2's conjunction according to the registered rule (with the exact permutation p for the rank component), and it no longer overwrites existing `RESULTS.*`: a re-run writes to `RESULTS-RERUN-<time>.*` instead. Run on a copy of the confirmatory files, it reproduced the verdicts and Holm values obtained under the registered conjunction rule. The frozen `RESULTS.*` are kept as they are. | Round-3 review, leg B [omitted: internal tool name] |

## D6 correction (2026-10-24, after the round-8 review; the text above is left unchanged)

The reason D6 gives is "only at τ_slow = 80 are both time scales visible in the extrapolation window". **This sentence is wrong.**

- In the extrapolation window [51, 200], the deviation of the metastable Bayes prediction from the marginal distribution decays as a **single exponential** whose time constant is τ_slow: the ratio between adjacent steps equals e^{−1/τ_slow} to within 10⁻⁵ (for both τ_slow = 20 and 80; recomputed by the author, seed 777, 2,000 sequences).
- The fast observable mode (decay time 2.61 or 2.89) has decayed to about e^{−17} by k = 51 and is not visible in the window.
- Hence on metastable, H3 tests "relaxation at a single rate", which is exactly what the pure contraction control C parameterises directly. H3's premise that "two time scales cannot be captured by a single contraction" cannot show up in the extrapolation window; it is visible only within the training range.
- The frozen verdict is unchanged (H3 NOT MET, in the opposite direction); what changes is the interpretation of this result, which report revision 8 has written down.

## D14(f) (2026-10-24, recorded after the round-8 review)

In T1, E1 and E2, the HMM is fitted only on the first 5,000 training sequences or windows (`run.py:157`, `run_e2.py:111`, `run_ethucy.py:118`), whereas the training set specified by the frozen texts is all sequences (v1 §2: 20,000 sequences; E2-DESIGN §1: 20,000 per seed; E2.1 §2: every window is used). Consequences: in E1 the truncation follows scene order, so four of the five folds lack entire scenes; in E2, B2 and B3 cover only the earliest 42% and 81% of the training period. Report §3.7, §3.8 and the limitations section state this as it is; all HMM comparisons are confounded by it.

# E3 deviation records

## D15 — The reason for keeping A3 was registered after the freeze (2026-10-24, recorded after the freeze)

- E3-DESIGN §0 requires that A3 (the d = 4 qudit clock), which is not triggered by any registered failure, be demoted to descriptive item D8 [Translator's note: D8 here is the E3 descriptive item D8 of E3-DESIGN §6, not deviation D8.], and that "the reason for keeping it is written into DEVIATIONS.md as a D15+ entry, registered before the freeze". **This entry was overlooked before the freeze**; the omission was found by the independent verification after the freeze (at `590127b3`, DEVIATIONS.md went no further than the D14 supplements).
- Reason (written retrospectively, following the original intent of design §0): to examine how the moment-matching (method-of-moments) repair behaves as the dimension increases (d = 4, O = 4, relaxation 20 steps, built by the same construction as A2); descriptive only, carrying no decision rule.
- Impact: none. G9, following the registered order, has already trimmed A3 (TRIMS contains "A3"), and the confirmatory phase runs no A3 job.

### D15 correction (2026-10-30, recorded after the round-11 claim review [omitted: internal tool name]; the text above is left unchanged)

- The statement above that "the confirmatory phase runs no A3 job" is inaccurate: the confirmatory phase ran 10 A3 reference-predictor jobs (`runs/e3/e3-U-A3-REF-REF-s20…s29.json`) and the A3 fixture checks; what G9 trimmed were A3's model arms (HQ(4)-M and C2).
- At the development-gate stage (seeds 200–202; G7 descriptive only), M and O05 were run on A3: M's CF was 0.683, 0.309 and 0.460, and O05's was about 0.987 on all three (`runs/e3/e3-DEV1-A3-HQ4-*`). Technical report §4.4 reports these as a descriptive item.

## D16 — Confirmatory compute may exceed the 32 CPU-hour hard cap (2026-10-24, recorded after the freeze)

- G9's projection used CPU measured on the development set: after the 7 trims, development plus confirmatory came to 29.8 CPU hours (cap 32).
- The independent verification after the freeze measured that, under the current machine load (load average 14–22, 10 cores), the `cpu_s` of confirmatory jobs is about 1.27 times the development value (M 808 against 602 s, C1 842 against 651 s, S 832 against 686 s). Extrapolating on this basis, development plus confirmatory may come to about 36.5 CPU hours, above the hard cap.
- The design forbids any further trim or design change after the freeze, so **nothing is changed** and the runs proceed as normal; after they end, the actual CPU and this entry are reported in the summary.
- Secondary consequence: H6's compute check (median `cpu_s` of each cell of the decisive ladder ≥ 0.8 × the median of HQ3-M on A2) may FAIL because of the difference in load; H6 is already a descriptive item, and this does not affect any registered verdict.
- Also noted: apart from C0, the records do not record a commit ID separately; instead they bind the code through `code_manifest_sha256` (628a28ec…, equal to the value recomputed on the tree of commit `590127b3` and to the timestamped manifest); this complies with design §3 and §16.

### D16 closure (2026-10-25, recorded after E3 ended)

- Actual CPU: confirmatory jobs 24.45 CPU hours (`job_cpu_s`; 24.22 by `cpu_s`) + development gate 4.71 = about 29.2 CPU hours, below the 32-hour hard cap; the roughly 36.5 hours extrapolated in this entry did not occur (the two independent verifications [omitted: private file path] agree). [Translator's note: these internal verification records are not part of this supplement; technical report §4.4 summarises the verification.]
- The compute check FAILed, as this entry anticipated (the 6 cells of the decisive ladder are at 0.66–0.79 times the HQ3-M median of 765 s); this affects only H6, which is already a descriptive item.
- The `e2_code_commit` in the C0 records is the HEAD at the start of each job: 38 of the 40 are `7a3bc627`, an unrelated data commit that landed during the run, and 2 are `590127b3`. It is not the code identity; the code binding is given by `code_manifest_sha256` (`628a28ec…`), and recomputation on the current working tree still matches.

---

[omitted: unpublished follow-up experiment]
