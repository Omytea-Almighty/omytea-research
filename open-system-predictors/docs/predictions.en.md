# Predictions written before the results (T1 and E3 blocks, with reconciliations): excerpt (English version)

Original: PREDICTIONS-BEFORE-RESULTS.md (Chinese, not published). SHA-256 of the original: f33440c11448a9cba6b5757010addb701000f56ce6e5732168ae6dd851ce945b (the whole file as it stood on 2026-10-30, including the parts not translated here). The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). Its only time evidence is the private git history. The file was first committed to the private repository at 2026-10-22 10:42:40 UTC (git), while its own first heading gives 10:45 UTC. Later parts were appended in order: the reconciliation at 2026-10-22 20:46:32 UTC, the reconciliation correction at 2026-10-23 09:40:08 UTC, and the E3 block at 2026-10-24 15:59:35 UTC (git). Lines 1–58 of the original, the part translated here, are byte-identical to the version committed at 2026-10-24 15:59:35 UTC (SHA-256 8e21f677554f470566b90bb128f80945decc2543c3b901269b9c0efe6fcde984). The E3 block's predicted probabilities are reproduced in the frozen E3 summary code (`src/osw/summary_e3.py`, constant `PREDICTIONS`; the original describes this as verbatim), keyed by the code's label constants. For PC-B, PC-K, H6 and STOP the constant also stores the complementary label (FAIL 0.05, FAIL 0.15, NOT_HOLDS 0.25, NOT_TRIGGERED 0.97); it does not store the reasons column. That code file (SHA-256 a7ca632740bab29d81e86bd9b37cfe68b47568b96ff5e695a953f770af5e0455) is listed in the E3 code manifest (`docs/timestamps/2026-10-24-e3-design/E3-CODE-MANIFEST.json` in the supplement's `repro/` folder; file SHA-256 118501c04f25062fcc31cfc94f78e514c13550ec6beeafdb08339caf40a35aee; aggregate hash recorded inside it 628a28ec6c35ec1248a257837514486b1f65695e1f173be92daff62ebe10c340), whose file hash carries RFC 3161 tokens from DigiCert (2026-10-24 15:57:57 UTC) and Sectigo (2026-10-24 15:57:59 UTC). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

This is an excerpt: it translates the T1 prediction block with its reconciliation and reconciliation correction, and the E3 prediction block (lines 1–58 of the original); the remaining parts of the original are marked as omitted in place. The final section, "E3 reconciliation", is translated from §6 (lines 94–114) of E3-RESULTS.zh.md (Chinese, not published; SHA-256 320339e864acc1991ff48d7586b9250d3124ce09c6aaa11e6402e67c3cbc9e31; written after the E3 confirmatory results; not committed to the private repository as of 2026-10-30; no independent timestamp). In the E3 parts, outcome labels are rendered as the label constants of the frozen E3 summary code (for example NOT_REACHED, LOCAL, ESTABLISHED, UNCERTAIN); the T1 parts keep the labels PASS / NOT MET used in the original.

---

# Predictions written before the results (2026-10-22 10:45 UTC, confirmatory phase at 181/450, the author has not looked at any confirmatory data)

[Translator's note: The time is as written in the original. The file's first git commit is at 2026-10-22 10:42:40 UTC.]

**Basis:** only the results of development seeds 101 and 102, and structural facts. These predictions are written so that, once the results are out, they can be reconciled against them to calibrate judgement; they do not affect any analysis.

| Item judged | Predicted PASS probability | Reason (development data) |
|---|---:|---|
| H1 simple τ=20 | 0.45 | U2 − L extrapolation KL: about 0.030 on seed 102, about 0.015–0.018 on seed 101. The mean may land near 0.02; it must also pass the Holm correction and the two intervals |
| H1 metastable τ=20 | 0.15 | About 0.017 < 0.02 on seed 102; the Bayes signal in this configuration's extrapolation window is only 6×10⁻⁵ |
| H2 | 0.10 | On seed 102 the extrapolation KL is not monotone in d_E: U1 0.032, U2 0.030, U4 0.014, U8 0.051; U8 is the worst |
| H3 | 0.05 | On seed 102 the extrapolation KL of C (0.00042) is lower than that of L1 (0.00286), the direction opposite to H3 |
| H4 simple / metastable / action non-inferiority | 0.90 / 0.90 / 0.75 | The extrapolation KL of L differs from the best classical baseline by less than 10⁻³, far smaller than the margin of 0.01 |
| KT2 triggered (H1 τ=20 difference < 0.01 and the CI spans 0) | 0.15 | — |

**Most likely overall picture:**
- Beyond the training horizon the unitary predictor produces excess structure, i.e. H1-simple may hold or nearly hold;
- but the mechanism attribution "finite-dimensional recurrence" (H2) and "beyond ordinary contraction" (H3) most probably do not hold;
- what helps is "contraction towards the stationary state" itself, not the general structure of GKSL.

## Reconciliation (2026-10-22 20:45Z, appended after the confirmatory results were out; the text above was not changed)

| Item judged | Predicted probability | Result |
|---|---:|---|
| H1 simple | 0.45 | PASS |
| H1 metastable | 0.15 | NOT MET |
| H2 | 0.10 | PASS by the letter of the pre-registration (exactly at the boundary); the frozen code judged NOT MET |
| H3 | 0.05 | NOT MET (opposite direction, consistent with the prediction) |
| H4 ×3 | 0.90 / 0.90 / 0.75 | all non-inferior |
| KT2 triggered | 0.15 | not triggered |

- **Brier score** (8 items; 0.25 if 0.5 is reported for every item): 0.155 counting H2 by the letter of the pre-registration, 0.055 counting it by the frozen code.
- **The largest deviation:** I gave H2 only 0.10, and the result passed at the boundary.
- **The overall picture agrees with the advance prediction:** the phenomenon holds (H1 simple), the mechanism attribution does not (H3 in the opposite direction, H2 holds only at the boundary), and contraction by itself is enough.

## Reconciliation correction (2026-10-23, after the round-2 review; the text above is kept unchanged)

- Under the registered conjunction rule, the outcome of H2 is NOT MET (see the D13 correction in DEVIATIONS.md). I had given it a pass probability of 0.10 in advance, so this item was **essentially predicted correctly**, and is not the "largest deviation" stated above.
- Counting this outcome, the Brier score of the 8 predictions is 0.055 (0.25 if 0.5 is reported for every item). The 0.155 computed above with "H2 PASS by the letter of the pre-registration" is void.

# Predictions written before the E3 results (2026-10-24, before freezing; the author has not seen any confirmatory data, seeds 20–29 not run)

**Basis:** the exploratory diagnosis of 2026-10-24 (seeds ≥ 300, independently reproduced) and the results of the E3 development set (seeds 200–204); saved verbatim at freezing by `src/osw/summary_e3.PREDICTIONS`, with Brier scores computed after the runs end. The numbers given by the design group in design §6 are estimates made before the development results were seen and are kept as history; **this block is the one registered and scored.** H6 and STOP were demoted to descriptive items before freezing (the G8 planted-truth simulation did not pass the gate); their predictions are only recorded, not scored.

[Translator's note: In this block and in the E3 reconciliation below, H1–H7 and STOP are the E3 rules that the technical report calls E3-H1 to E3-H7 and E3-STOP; they are not the T1 hypotheses H1–H4 above. HQ2-M, OOM3 (in H7) and HMM4-C are written HQ(2)-M, OOM(3)-C and HMM(4)-C in the report.]

| Item judged | Predicted probabilities | Reason (development set or exploratory evidence) |
|---|---|---|
| H1 (≥ 9/10 seeds with M ≥ 0.90 on A2) | PASS 0.85 · PARTIAL 0.12 · KILL 0.03 | Development 3/3 at about 0.993, exploratory 7/7 at about 0.99; the risk is that the moment fit occasionally lands in another basin |
| H2 (Stiefel optimiser S reaches the threshold) | NOT_REACHED 0.85 · PARTIAL 0.12 · REACHED 0.03 | In development about 0.17–0.20 at τ0 = 0.1, negative at the other τ0; the confirmation uses 6 restarts, possibly slightly better |
| H3a (harmlessness test on A1) | PASS 0.92 · PARTIAL 0.06 · KILL 0.02 | On A1, M 0.998–0.999, O05 the same |
| H3b (transfer to the weak-signal A2w, judged when H1 PASS) | PASS 0.03 · PARTIAL 0.08 · KILL 0.85 · STAT_LIMITED_NOT_JUDGED 0.04 | In development M 0.01–0.30, O05 about 0.99: the fix most probably does not transfer to a process whose signal is only 0.003 nats per step |
| H4 (basin locality, fraction with CF ≥ 0.90 at σ = 0.3) | LOCAL 0.92 · UNDETERMINED 0.06 · NONLOCAL 0.02 | Development, 3 seeds at σ = 0.3: CF −0.003, 0.01, 0.26 |
| H5 (attribution to the warm start) | ESTABLISHED 0.92 · UNDETERMINED 0.06 · REFUTED 0.02 | Development C1 (3 restarts) 0.06–0.19, C0 0.02–0.06; the confirmation uses 6 restarts |
| H7 (HQ2-M and OOM3 equivalent on A1, ±0.03) | EQUIVALENT 0.88 · HQ_BETTER 0.02 · OOM_BETTER 0.02 · UNCERTAIN 0.08 | Development HQ2 0.998–0.999, OOM3 0.991–0.998 |
| PC-B (HQ2-M − HMM4-C ≥ 0.10) | PASS 0.95 | Development difference about 0.20 (0.999 vs 0.79–0.80) |
| PC-K (the classical pipelines reproduce known levels) | PASS 0.85 | A1 GRU4 0.985–0.989, LRU16 0.915–0.937; on A2 the small-scale cells are not below E2; GRU16/LRU16 not tested in development |
| PC-C (the E2 protocol, unchanged, reproduces the stall; modifier) | REPRODUCED 0.95 · PARTIAL 0.04 · NOT_REPRODUCED 0.01 | Development C0 0.02–0.06 |
| H6 (descriptive item, recorded, not scored) | mechanical label HOLDS 0.75 | In development GRU32 0.55 and LRU32 0.80 are still below 0.90; TOP32 has been trimmed, the ladder stops at GRU16/LRU16, possibly right-censored |
| STOP (descriptive item, recorded, not scored) | STOP_TRIGGERED 0.03 | — |

**Most likely overall picture:** the moment-method fix holds on new seeds (H1 PASS, H5 ESTABLISHED), the E2 stall is reproduced (PC-C), the literature-default Stiefel optimiser does not reach the threshold with this implementation and compute (H2), but the fix does not transfer to the weak-signal A2w (H3b KILL).

---

[omitted: unpublished follow-up experiment]

---

[omitted: internal administrative note]

---

# E3 reconciliation

Translated from §6 (lines 94–114) of E3-RESULTS.zh.md (see the header).

## 6. Prediction reconciliation and Brier scores

The registered, scored predictions are the E3 block of `PREDICTIONS-BEFORE-RESULTS.md` (written by the author before freezing, without having seen confirmatory data, with reference to the development set; verbatim identical to `summary_e3.PREDICTIONS`, checked). Multiclass Brier (sum of the squared errors over the classes):

| Rule | Actual | Registered prediction (probability of the actual class) | Brier (registered) | Brier (original design §6 numbers, for comparison only) |
|---|---|---|---|---|
| H1 | PASS | 0.85 | 0.038 | 0.140 |
| H2 | NOT_REACHED | 0.85 | 0.038 | 0.335 |
| H3a | PASS | 0.92 | 0.010 | 0.038 |
| H3b | KILL | 0.85 | 0.031 | 0.935 |
| H4 | LOCAL | 0.92 | 0.010 | 0.140 |
| H5 | ESTABLISHED | 0.92 | 0.010 | 0.065 |
| H7 | UNCERTAIN | **0.08** | **1.622** | 1.188 |
| PC-B | PASS | 0.95 | 0.005 | 0.020 |
| PC-K | PASS | 0.85 | 0.045 | 0.020 |
| PC-C | REPRODUCED | 0.95 | 0.004 | 0.035 |
| **Mean** | | | **0.181** (0.021 without H7) | 0.292 (0.192 without H7) |

- The "most likely overall picture" written in the prediction block was borne out in full, except for H7.
- **The only large miss is H7:** on the development set HQ2 was 0.998–0.999 and OOM3 0.991–0.998, on which basis the author gave EQUIVALENT 0.88; on the confirmatory seeds OOM3 was more dispersed (minimum 0.885), and OOM fine-tuning was generally below its spectral initialisation, so the interval of the difference shifted outside the equivalence window. Lesson: 3 development seeds underestimated the seed-to-seed dispersion and the optimisation degradation on the baseline side; [omitted: unpublished follow-up experiment]
- The original design §6 numbers (made before the development results were seen) were clearly too optimistic on H3b (only 0.20 for KILL); the development-set revision corrected this to 0.85.
