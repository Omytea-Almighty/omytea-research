# Open-system world model: development-gate diagnosis (read-only, 2026-10-19), summary (English version)

Original: GATE-DIAGNOSIS.zh.md (Chinese, not published). SHA-256 of the original: a97c78831e72cf1bc8f9312b7e9834b6e705d94f653c9be65d17db9783c36735. The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). It has not been committed to the private repository as of 2026-10-30, so there is no repository time evidence either; its dates rest on its own text (snapshot 2026-10-19T15:40:55Z; follow-up section 15:55 UTC) and on the file's modification time (2026-10-19 15:55:59 UTC). This document is a summary of that internal diagnosis document, not a translation. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

[Translator's note: D01–D06 are the diagnosis's own finding labels; the technical report cites finding D02 in this form, and the theory notes cite D04 and D05. They are not the deviations D1–D16 of DEVIATIONS.md. Where the diagnosis refers to a deviation, this summary writes "deviation Dn".]

[Translator's note: File and line references are to the source as it stood at the diagnosis snapshot (2026-10-19T15:40:55Z). Several of these files were changed before the confirmatory runs (DEVIATIONS.md D5–D9, D11), and `analyze.py` was changed again after them (D13, D14).]

[Translator's note: The v1 confirmatory phase was never run. The confirmatory runs in the technical report were made under the v1.1 pre-registration (DEVIATIONS.md D10; report §4.1, seeds 0–9).]

## Status and conclusion

Status: INTERNAL PRACTICE / GATE FAIL PRESERVED / CONFIRM NOT STARTED; snapshot 2026-10-19T15:40:55Z. Conclusion: the confirmatory phase cannot be released. Besides the two best-learning-rate configurations that still fail, the diagnosis found substantive problems in task identifiability, in the theory's quantitative bounds and in the statistical reporting. It describes itself as an internal diagnosis, not a theory certification, not a passed external review and not evidence that the work is ready for submission to TMLR.

## Scope and method

The diagnosis read the current source code, the 48 development results, the original gate, the logs, the pre-registration and the theory draft, and ran one read-only NumPy/standard-library diagnostic (0.0318 s). It loaded no model weights, did no training, did not run the original gate (which would overwrite `GATE.md`), did not start the queue, and did not modify the experiment, the thresholds, the confirmatory seeds or the original pre-registration. Its only outputs were three files in its own directory. All sources come from the same project; the diagnosis does not claim three independent source families. Its theoretical counterexample is an analytic derivation made in the diagnosis and still needs independent mathematical review. A forensic machine-readable record [omitted: private document] holds the per-result SHA-256 values, source pins, the check of the original payloads, the numerical values and the full development table.

## D01: 7 trials failed, not 7 models; at the best learning rate 2 combinations still FAIL

The original gate threshold, ln(4) − 0.003 = 1.383294361119891 nats, is checked per trial, not after learning-rate (LR) selection (`src/osw/gate_dev.py:18`). The 48 results are 42 training trials plus 6 Bayes/HMM references, and all 48 payload digests are valid. There are 21 training combinations with 2 LRs each; 7 trials failed, involving 5 (task, model) combinations. Taking each combination's minimum validation NLL, 19 of 21 lie below the threshold and two still fail: simple/U1 (LR .003, 1.386539220810; .001 also failed) and metastable/U2 (LR .003, 1.386261582375; .001 also failed). simple/U2, simple/U4 and metastable/C failed only at the lower LR. The 7 trial failures may not be escalated into "all quantum / all unitary models are unlearnable", and the original gate may not be lowered, nor the aggregation changed, to declare PASS. The original log records an explicit stop at 11:20:30 EDT, and `runs/confirm` did not exist. [omitted: internal administrative note] Evidence: `runs/dev/GATE.md`; `runs/chain.log` line 10; `src/osw/queue.py:36`. Best LR per combination, as given in the diagnosis:

| task | model | best LR | validation NLL (nats) | diagnostic | raw parameters | dev extrap KL |
|---|---|---:|---:|---|---:|---:|
| action | C | 0.001 | 1.349880457 | below threshold | 897 | 0.0003311777 |
| action | GRU | 0.003 | 1.362558126 | below threshold | 1268 | 0.0002703225 |
| action | K | 0.003 | 1.348350644 | below threshold | 1536 | 0.0003647232 |
| action | L | 0.001 | 1.349399090 | below threshold | 1152 | 0.0003839855 |
| action | LRU | 0.003 | 1.364893436 | below threshold | 1236 | 0.0004075059 |
| action | U2 | 0.003 | 1.355919719 | below threshold | 1536 | 0.004584117 |
| metastable | C | 0.003 | 1.381948709 | below threshold | 769 | 0.00004434110 |
| metastable | GRU | 0.003 | 1.381448150 | below threshold | 1220 | 0.00005790899 |
| metastable | K | 0.003 | 1.381489038 | below threshold | 1024 | 0.00004682087 |
| metastable | L | 0.003 | 1.381617665 | below threshold | 1024 | 0.000004136119 |
| metastable | LRU | 0.001 | 1.381593466 | below threshold | 1188 | 0.00001306559 |
| metastable | U2 | 0.003 | 1.386261582 | FAIL | 1024 | 0.0002170276 |
| simple | C | 0.003 | 1.349395037 | below threshold | 769 | 0.0001474372 |
| simple | GRU | 0.003 | 1.322183609 | below threshold | 1220 | 0.0001102748 |
| simple | K | 0.003 | 1.348226666 | below threshold | 1024 | 0.0002261449 |
| simple | L | 0.003 | 1.349132180 | below threshold | 1024 | 0.0001993444 |
| simple | LRU | 0.003 | 1.323450923 | below threshold | 1188 | 0.0002119315 |
| simple | U1 | 0.003 | 1.386539221 | FAIL | 640 | 0.0003960848 |
| simple | U2 | 0.003 | 1.357060909 | below threshold | 1024 | 0.02217509 |
| simple | U4 | 0.003 | 1.352283239 | below threshold | 2560 | 0.02099870 |
| simple | U8 | 0.001 | 1.352593780 | below threshold | 8704 | 0.01676633 |

## D02: the metastable task's slow mode is unobservable, so the premise of H3's mechanism discrimination fails

In the source, B[s,:] = B[s+4,:], and the two 4-state blocks share the same W: P = a diag(W,W) + (1−a) J8, with a = exp(−1/20), W = b C4 + (1−b) J4 and b = exp(−1/3). With F = [I4; I4], PF = FQ holds exactly, where Q = aW + (1−a) J4 = ab C4 + (1−ab) J4, and B = F B4. The slow block-difference mode v = (1,1,1,1,−1,−1,−1,−1) satisfies vP = av but vB = 0. The whole observation process (not only some marginal) therefore lumps to a 4-state single-rate rotation-plus-relaxation process with τ_eff = (1/3 + 1/20)^−1 = 2.608695652174. Context cannot recover the block label hidden by the symmetric emission, and the source comment that the order "carries information about s and s+4" does not hold for the current symmetric cycle. An independent matrix check gave max|PF − FQ| = 1.1102230246251565e-16, error 0 for the single-rate Q identity, and max|vB| = 1.1102230246251565e-16 (`src/osw/tasks.py:22`, `src/osw/tasks.py:54`). A slow latent eigenvalue does not imply two identifiable decay rates in the observed predictions, so H3 currently cannot discriminate "generic dissipation beyond ordinary contraction", and a difference between the two models could be only fitting or optimisation. At k = 51 the observable signal has decayed as exp[−51(1/3 + 1/20)], and the uniform prediction is already extremely close to Bayes. The existing Bayes NLL over the first 50 steps on the test set is 1.382907014218; ln 4 minus this is about 0.003387, only 0.000387 from the gate, and the extrapolation NLL is ≈ ln 4. These are test samples, not the gate's first 1,000 validation samples, and may not be used directly to judge the validation optimum or to change the threshold. Recommendation: fix the observability of the mechanism task first (training is not the first step); a new task must be a separate version, frozen again and labelled exploratory under §5 of the original pre-registration, and it does not replace the v1 results.

[Translator's note: Following this finding, the metastable emission was changed before the confirmatory runs (DEVIATIONS.md D6 and the D6 supplement; report §3.1). DEVIATIONS.md also records a later correction (D6 correction, 2026-10-24): with the revised emission, the Bayes predictive's deviation from the marginal in the extrapolation window [51, 200] decays as a single exponential with time constant τ_slow (the fast observable mode, with decay time 2.61 or 2.89, has vanished by k = 51), so H3's two-timescale premise cannot show in the extrapolation window and is visible only within the training range.]

## D03: validation NLL ≈ ln 4 cannot be attributed to "the task is unlearnable" or to "the theoretical representation failed"

On the simple task, GRU, LRU, L, K, C and U2/U4/U8 all show a validation learning signal, so the current data refute a global "the task is unlearnable"; the metastable task only has a very small average information margin and is not strictly unlearnable. The loss is the mean natural-log NLL over batch × 50 horizons, in nats per prediction, not a 50-fold total-NLL unit error (`run.py`, lines 96–114). Its expected objective equals that of sampling k uniformly, but all 50 steps are actually computed, and this implementation difference must be disclosed. The existing results contain only the current-batch training loss every 200 steps, with no initial validation, gradient or parameter changes, per-horizon validation or saved weights, so stuck optimisation, poor initialisation and insufficient representation cannot yet be told apart. That changing the LR rescues 3 combinations is evidence of optimisation sensitivity, not a unique causal explanation. The formal seeds 0–9 were still unused, and development seed 100 may not be entered into the H1–H4 confirmatory statistics.

[Translator's note: The v1.1 pre-registration (§4) discloses the all-50-horizon loss; deviation D11 and v1.1 §8 added the initial validation NLL, gradient norms and parameter checkpoints to the run records.]

## D04: the theory draft's dimension-uniform "recurrence-interval bound" is false

THEORY-DRAFT.md (line 21) mistook Dirichlet's first near-return bound for a gap bound under which every late window contains a near-period. Counterexample: fix D = 2 and ε = 0.1, U = diag(1, exp(2πi/N)), ρ = |+⟩⟨+|, and a POVM measuring |+⟩⟨+| and its complement, so that p_k = ((1+cos(2πk/N))/2, (1−cos(2πk/N))/2), with N any large multiple of 6. For every integer τ ∈ [N/6, N/3], ||p_τ − p_0||_1 = 1 − cos(2πτ/N) ≥ 1/2 > ε, so this interval, of arbitrarily large length N/6, contains no ε-almost-period; with D and ε fixed, the claimed upper bound on the relative-density gap (the inclusion length) that depends only on D and ε does not exist. Relative density for a fixed given U, and the early–late trade-off via the triangle inequality, may be kept as qualified propositions awaiting independent proof, but ℓ must depend on the spectrum or the dynamics, and one random 16-dimensional numerical sample cannot confirm a proof. A larger dimension also does not guarantee slower return or that the extrapolation KL necessarily strictly decreases: extending U to U⊗I with the readout unchanged leaves the whole prediction sequence unchanged (this constant sequence alone cannot refute weak monotone non-increase). H2 remains an empirical hypothesis. Even a lower bound on the error at "some step" would not automatically give a mean KL ≥ 0.02 over [51, 200], because sparse bad points can be diluted by the average, and conclusions for a fixed U do not cover products U(a_k)…U(a_1) for arbitrary aperiodic action sequences.

[Translator's note: THEORY-DRAFT.md was corrected accordingly on 2026-10-21; the retracted bound is kept there struck through.]

## D05: classical inclusion in CPTP does not mean that the current L/K classes represent the task exactly

The constructive algebra of the draft's Proposition 2 reads correctly for d_S = S. For d_S > S, the sums of the listed Kraus operators and of the instrument give only the projector onto an S-dimensional subspace, and a valid extension on the orthogonal complement is needed for trace preservation on the whole space. More importantly, a general discrete CPTP classical embedding is not guaranteed to have an exp(GKSL) realisation with the current three jump operators, nor to fall inside the rank = 4 K parameterisation. Existence over all CPTP families cannot be restated as a zero-error theorem for the current L, nor as a guarantee that training will find those parameters (THEORY-DRAFT.md line 43; `src/osw/models.py:156`, `src/osw/models.py:196`).

[Translator's note: THEORY-DRAFT.md adopts this finding as its list item 5; in this experiment d_S = S = 8. For T1-simple specifically, later internal reviews found exact predictors within the registered three-jump L and rank-4 K families (report §3.2).]

## D06: reporting and comparison boundaries to fix before confirmation (not changed by the diagnosis)

- In `analyze.py`, the PASS decisions at lines 67/77/83 do not depend on the Holm correction at line 97; intervals and a PASS can be produced from only 2 seeds, and the full set 0–9 is not enforced. No confirmatory results existed, and the diagnosis did not run the script to produce conclusions.
- For H4 the best classical model is selected by confirmatory test KL (lines 88–92), a selection bias; "parity_or_better" checks only that the point estimate is ≤ 0.01, with no equivalence test; the Holm dictionary has no H4. The code could not be described as "strictly following the pre-registered statistics".
- H1 raw parameters: on simple/metastable L = U2 = 1024; on action L = 1152 and U2 = 1536, a 33.3% difference (relative to L). C, GRU and LRU also exceed ±10% in several places, and no second, parameter-matched set was found. Parameter counts are not effective degrees of freedom or FLOPs.
- `queue.py:58` gives U1/U4/U8 the LR selected for U2, not each model's own best LR; U8's best LR in this run was .001, but the code would use U2's .003. This needs independent clarification and must not be changed silently.
- The theory's "late worst-case window", the experiment's "mean KL" and the peak sum of deviation D4 (the revival index: peaks of D(k) measured against the empirical mean over k ∈ [400, 600]) are not the same object; the peak sum alone cannot distinguish genuine recurrence from generic oscillation.
- On the single development seed, simple U2 − L extrapolation KL ≈ 0.021975743 may be recorded as an exploratory candidate; metastable ≈ 0.000212891. Neither is an H1 confirmation or a formal kill-test conclusion.

[Translator's note: Before the confirmatory runs, deviation D8 (which adopts this finding) changed the analysis code for the first two items and deviation D9 realigned the parameter budgets of GRU and LRU and added L1, parameter-matched to C, for the third (the action-task U2 and K, 1,536 against L's 1,152, were left unmatched and enter no hypothesis); under the v1.1 pre-registration (§4, deviation D5) the LR is selected separately for each (task, τ, model) on development seed 102.]

## Proposed next step (not executed in the diagnosis)

The diagnosis proposed first validating a new task specification with a training-free observability positive/negative control, keeping the v1 originals unchanged. In a separate candidate (candidate A), B' = B + 0.05 v cᵀ with c = (1, −1, 0, 0): row sums stay 1, the smallest entry is 0.05, the stationary observation stays uniform, and vB' = 0.4c makes the slow mode visible. The original B is kept as the negative control, and both visible non-stationary eigenmodes are to be checked, so that the change is not merely a renaming. The diagnosis gives a short NumPy small-matrix check for this (no model, no training, no writes; the listing is not reproduced in this summary); the B' experiment was not run. A passing candidate would still need an external mathematical check and a new task and analysis pre-registration, and the original confirmatory phase was not to be started directly. A same-budget rerun of simple/U1 on development seed 101, with added gradient and initial-validation diagnostics, could then separate initialisation faults from training faults; exploratory results may not be back-filled into the original confirmatory series.

## Independent follow-up by the commissioning task (15:55 UTC)

Candidate A remains the diagnosis author's unexecuted proposal. The task that commissioned the diagnosis separately completed a small-matrix check of a different candidate (candidate B) [omitted: private document], with the increment v(.06, .02, −.02, −.06): minimum probability .04, row sums 1, uniform stationary readout, vB' = (.48, .16, −.16, −.48). This shows that the candidate makes the slow mode visible in the emission; it does not show matched statistical difficulty of the new task, full identifiability, effective training or any model advantage. A re-check by a reviewer other than the diagnosis author [omitted: internal administrative note] independently confirmed, for the original task, PF = FQ, the per-symbol identity D_o F = F D_{o,4} and the lumping of observation words, supporting the equivalence of the whole observation process; it also confirmed τ_eff = 60/23 and the two-dimensional slow-frequency counterexample. Relative density for a fixed U is not overturned by that counterexample. Following that review, the diagnosis narrowed "monotone decrease" to "necessarily strictly decreases" (D04). The old theory, task, gate and pre-registration were not changed.

[Translator's note: Neither candidate A nor candidate B was adopted; the v1.1 task distinguishes the block identity by symbol pairs (DEVIATIONS.md, D6 supplement).]

## Not covered, counter-evidence and handoff

The diagnosis did not rerun training or tests, did not check whether training processes were alive, did not complete the three-source external literature verification or the novelty review, did not verify the full theoretical proofs, and did not check all numerical stability, random seeds or optimisation paths. Citing the current source shows what the implementation does, not the authors' independence or any external recognition. The formal significance, correctness and artifact-truth gates and the two-round external review gate were not passed by this material.

[omitted: internal administrative note]
