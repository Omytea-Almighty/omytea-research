# Adjudication of the v1.1 confirmatory results (English version)

Original: RESULTS-ADJUDICATION.md (in `runs/confirm/` of the private repository; Chinese, not published). SHA-256 of the original: 6b1e92a111a47e9d53ca734c7c64a8cb4ca0c9962395af6c2d3e7f3731be8166. The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). Its only time evidence is the private git history: the text before the correction section was first committed to the private repository at 2026-10-22 20:46:32 UTC (git), while its own heading gives 20:45Z; the correction section was added at 2026-10-23 09:40:08 UTC (git), giving the version hashed above. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

# Adjudication of the v1.1 confirmatory results (2026-10-22 20:45Z)

**Integrity checks (completed before the results were read):**
- Code manifest `4036f2e7…`: every file matches (`shasum -c` gave no output);
- 450/450 results, 45 for each of seeds 0–9;
- the `prereg_v11_sha256` of every result equals the hash of the frozen version, `d0199a7a…`.

**Raw output:** The output of the frozen analysis code is kept unchanged in `RESULTS.md` and `RESULTS.json`. This file adds only one adjudication: the floating-point issue in H2; see DEVIATIONS.md D13.

| Adjudication | Result | Basis |
|---|---|---|
| **H1 simple τ=20** | **PASS** | U2 − L extrapolation KL mean 0.0237 nats; bootstrap CI [0.0217, 0.0260], t CI [0.0211, 0.0264]; Holm p = 1.6×10⁻⁸ |
| **H1 metastable τ=20** | **NOT MET** | Mean 0.0189, below 0.02; the CI [0.0163, 0.0212] does not cross 0, but the point estimate does not reach the threshold |
| **H2** | **PASS by the letter of the pre-registration (exactly at the boundary); frozen code output NOT MET (floating-point error)** | Extrapolation KL by d_E: 1 → 0.0322, 2 → 0.0241, 4 → 0.0199, 8 → 0.0239. The exact Spearman ρ = −4/5, which satisfies "ρ ≤ −0.8"; the code computed −0.7999999999999999 and therefore judged it wrongly. Revival index U2 − L = 4.26, CI [3.95, 4.59], Holm p = 4×10⁻⁹ |
| **H3** | **NOT MET (opposite direction)** | On metastable τ_slow=80, C − L1 = −0.0022, CI [−0.0033, −0.0011]: the pure contraction control is **better** than the parameter-matched Lindblad model. On simple the two are equivalent (TOST p = 7×10⁻¹⁶). Compared with the native-size L, the conclusion is the same |
| **H4 simple / metastable / action** | **All non-inferior** | L − GRU is +0.0001, +0.0015 and −0.0001 nats respectively, all far below the margin of 0.01; the strongest classical baseline was selected on development data, and for all three it is the GRU |

[Translator's note: The H2 verdict in this table is corrected to NOT MET in the correction at the end of this document (2026-10-23). The Holm p-values in the table are those of the frozen analysis code; the technical report's Table 1 also gives the Holm p-values under the registered H2 conjunction (e.g. 2.1×10⁻⁸ for H1 simple τ=20 and 1/3 for H2). DEVIATIONS.md D14(c) later records that no validation NLL was recorded for the HMM, so the H4 baseline selection was in effect between the GRU and the LRU only; the report gives a post hoc sensitivity analysis that includes the HMM as a candidate.]

**Robustness of H2 (must be reported together with the verdict):**
- ρ = −0.8 is the value for exactly one inversion among 4 points: the error at d_E = 8 is larger than at d_E = 4.
- The extrapolation KL at d_E = 8 and at d_E = 2 differs by only 0.00014 nats. If U8 were just slightly worse, ρ would become −0.4.
- So "the error decreases monotonically with the environment dimension" **is not substantively supported**: the error is lowest at d_E = 4 and rises again at 8. The verdict by the letter is PASS only because it falls exactly on the boundary.

**Stopping rules (kill tests) of the planning note:**
- **KT2 is not triggered.** The H1 τ=20 difference is 0.0237, greater than 0.01. The unitary model's extra error beyond the training horizon is measurable on these tasks.
- **KT3 is triggered.** The pure contraction control is not worse than CPTP; it is better. The dose–response relation also holds only at the boundary. Per the planning note, **the main conclusion can only be written as "contraction", not as "recurrence"**.

**One-sentence conclusion:** On these synthetic tasks and at a scale of about 1,000 parameters:
- the unitary predictor has about 0.024 nats more error beyond the training horizon than the open-system predictor (simple τ=20);
- what removes this error is "contraction towards the stationary state" itself; the general GKSL structure brings no additional benefit;
- the classical GRU and the Lindblad model perform comparably.

[Translator's note: A later correction (DEVIATIONS.md, D6 correction, 2026-10-24) records that in the extrapolation window [51, 200] the metastable task's Bayes predictive approaches the marginal at a single rate (time constant τ_slow; the fast mode has already decayed by k = 51), so H3 could not test its two-timescale premise there; the H3 verdict is unchanged. The technical report states the second point for the registered extrapolation windows only.]

## Correction (2026-10-23, after the round-2 two-model review; the text above is kept unchanged)

[Translator's note: The review named here is an internal review round, not external peer review; the technical report cites it as internal review r2.]

- **The H2 verdict is changed to NOT MET.** Under the conjunction rule of v1.1 §6 and D8, the p of H2 is the larger of "the rank-component p and the revival-component p". The exact one-sided permutation test of the rank component gives p = 1/6; after the conjunction the Holm-adjusted p = 1/3, so H2 does not hold. The verdict above, "PASS by the letter of the pre-registration (exactly at the boundary)", is wrong: at the time only the threshold comparison was corrected, and the conjunction test was not done. See the D13 correction and D14 in DEVIATIONS.md.
- The KT3 verdict is unchanged (still triggered), and the reason is more direct: the pure contraction control is not worse than CPTP, and H2 does not hold.
