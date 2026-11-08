# Pre-registration addendum v1.1: open-system (CPTP) versus unitary-evolution density-matrix world models (English version)

Original: EXPERIMENT-PREREG-v1.1.zh.md (Chinese, not published). SHA-256 of the original: d0199a7a4a9f74a5fef6d3f31abf7bec23d9a16cfd37e76eed8fbeebe8ad285c. The original is hash-committed: this hash carries RFC 3161 tokens from DigiCert (2026-10-21 20:08:10 UTC) and Sectigo (2026-10-21 20:08:11 UTC) (`repro/docs/timestamps/2026-10-21-prereg-osw-v1.1/`). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

**Status:** FROZEN, 2026-10-21. The round-2 development gate was passed on 2026-10-21 (`experiments/open-system-world-model/runs/dev2/GATE.md`); the hash of this file and its RFC 3161 and OpenTimestamps proofs are in `docs/timestamps/2026-10-21-prereg-osw-v1.1/`. After freezing, this file is not modified again; any later change can only be written as a new deviation record.
**Upstream:** v1 pre-registration (`EXPERIMENT-PREREG.zh.md`, SHA-256 `37172954…`, 2026-10-19T12:09:22Z). [Translator's note: Its English version is preregistration-v1.en.md in this supplement.] All clauses of v1 not changed here remain in force.
**Why v1.1 is needed:** v1 failed at its development gate (`experiments/open-system-world-model/runs/dev/GATE.md`); confirmatory seeds 0–9 were never used. The causes of the failure and the corrections are recorded item by item in D5–D10 of `experiments/open-system-world-model/DEVIATIONS.md`, and the read-only diagnosis [omitted: private file path] was adopted.

[Translator's note: A summary of this diagnosis is included in this supplement (development-gate-diagnosis-summary.en.md).]

**Labelling:** Under v1 §5, analyses after the changes below are "exploratory" relative to v1, and confirmatory relative to this v1.1. The technical report gives both labels side by side.

## 1. Hypotheses and thresholds (same as v1; only the tested objects change)

| # | Decision rule (thresholds unchanged) | Change in v1.1 |
|---|---|---|
| H1 | U2 − L in mean KL over the extrapolation window [51,200] ≥ 0.02 nats; tasks T1-simple and T1-metastable, τ = 20 | None. The emission matrix of metastable is changed per D6 |
| H2 | Spearman ρ ≤ −0.8 between the extrapolation KL for d_E ∈ {1,2,4,8} and d_E; the revival index of U2 is higher than that of L | None (T1-simple, τ = 20). The bound in the theory draft that "the larger the dimension, the slower the recurrence" has been withdrawn; H2 is only an empirical hypothesis |
| H3 | On metastable, C − L ≥ 0.01 nats; on simple, the two tie | metastable moved to τ_slow = 80 (D6); L replaced by the parameter-matched L1 (D9); "tie" replaced by a TOST equivalence test, margin ±0.01 (D8) |
| H4 | L is within ±0.01 nats of the strongest classical baseline, or better; report only | Tested on three tasks, simple, metastable and action, all at τ = 20; the baseline is selected by validation-set NLL on the development seeds; changed to a non-inferiority test, margin 0.01 (D8) |
| H5 | Approximate replication of 0.77 ± 0.05 on the UWM-JEPA hidden-velocity task | None (an approximate replication per D1; not blocking) |

[Translator's note: H5 was never run (technical report §3.6). In the H4 baseline selection, the HMM recorded no validation NLL, so in practice only GRU and LRU could be selected (DEVIATIONS.md D14(c)).]

## 2. Tasks (D6)

- **T1-simple, T1-action:** same as v1.
- **T1-metastable:** block A (states 0–3) emits s mod 2 with probability 0.7, block B (states 4–7) emits 2 + (s mod 2) with probability 0.7, and each remaining symbol with 0.1. The transition matrix is the same as in v1.
  - Observability check: at τ_slow = 20 the visible time constants are 2.61 and 19.99; at τ_slow = 80 they are 2.89 and 80.14.

  [Translator's note: A later correction (DEVIATIONS.md, D6 correction, 2026-10-24) found that in the extrapolation window [51, 200] the Bayes predictive of this task relaxes at a single rate (time constant τ_slow); the fast mode (2.61 or 2.89) has decayed by k = 51 and is visible only within the training range. The slow values 19.99 and 80.14 are τ_slow = 20 and 80 up to rounding (technical report §3.1). The registered verdicts are unchanged.]
- **Configurations run:** simple τ = 20, 80; metastable τ_slow = 20, 80; action τ = 20.
  - simple τ = 80 is a secondary result only.
  - simple τ = 5, listed in v1, is not run: no hypothesis depends on it.

## 3. Models and parameter counts (D9)

| Name | Size | Parameters, one action | Parameters, two actions |
|---|---|---:|---:|
| U(d_E) | d_E ∈ {1,2,4,8} | 640 / 1024 / 2560 / 8704 | 768 / 1536 / 4608 / 16896 |
| L | J = 3 | 1024 | 1152 |
| L1 | J = 1 | 768 | 896 |
| K | rank 4 | 1024 | 1536 |
| C | rotation + contraction towards a fixed state | 769 | 897 |
| GRU | hidden size 15 | 1099 | 1144 |
| LRU | state 20, readout 16 | 1004 | 1044 |
| HMM | 8 states, EM, 3 restarts | — | — |

- **All comparisons in the hypotheses satisfy ±10%:** U2 vs L (H1), C vs L1 (H3), GRU and LRU vs L (H4).
- **Comparisons that do not satisfy it but do not enter the hypotheses are reported as they are:** U2 and K vs L on the action task; and the dimension sweep of U4 and U8.

## 4. Training (D5)

- Optimiser Adam, batch 64, gradient-norm clipping at 5.
- **Steps:** 6,000 (v1: 2,000).
- **Learning-rate candidates:** {3e-3, 1e-2} (v1: {1e-3, 3e-3}). For each (task, τ, model), selected by validation-set NLL on development seed 102.
- **Basis:** probes on development seed 101, see DEVIATIONS.md D5. KL within the training range at 6,000 steps:
  - L: 0.0034 at learning rate 1e-2 (it stalls at 0.029 at 3e-3);
  - K: 0.0031 at learning rate 3e-3, 0.0019 at 1e-2.
- **Instrument initialisation:** near identity.
- **Loss:** multi-horizon NLL, averaged over all k = 1…50. v1 says "k sampled uniformly from 1–50"; the two have the same expectation, and the actual implementation is disclosed here.

## 5. Development gate (D7)

All of the following must hold before the confirmatory runs:
- all development jobs have results, and the queue log shows no failures;
- the validation NLL of L, L1, K, C, GRU and LRU at the selected learning rate is < ln 4 − 0.003;
- the validation NLL of U2 at the selected learning rate is < ln 4 − 0.003;
- U1, U4 and U8 are only required to have finite results;
- all unit tests pass.

If the gate does not pass, stop; do not enter the confirmatory phase.

## 6. Statistics (D8)

- 10 confirmatory seeds (0–9), each configuration run once; if any seed is missing, the verdict is INCOMPLETE.
- PASS requires all four of the following at once:
  - the effect-size threshold;
  - the bootstrap 95% CI does not cross 0;
  - the t-interval 95% CI does not cross 0;
  - Holm-corrected p < 0.05.
- **The Holm family has 7 members:** H1 (simple, metastable), H2, H3, H4 (simple, metastable, action). For a conjunctive hypothesis, the p-value is the maximum of the p-values of its components.
- The H3 contrast with the native-size L is a secondary result and does not enter the Holm family.
- Implementation and tests: `experiments/open-system-world-model/src/osw/analyze.py`, `tests/test_analysis.py`.

[Translator's note: The frozen `analyze.py` did not apply the conjunction rule above to H2: it treated the rank (Spearman) component as a fixed threshold without a p-value, so H2's entry in the Holm family carried only the revival component's p (DEVIATIONS.md D13, D13 correction, D14(a)). `analyze.py` was corrected after the confirmatory analysis (D14(e)); the frozen outputs are unchanged. The technical report gives both sets of Holm values; under the registered conjunction, H2 is not met.]

## 7. Unchanged parts

The following are all the same as in v1: the metrics, the windows, the revival index (D4), reporting all results, and the rules for external wording.

The E1 exploratory analysis follows the separately frozen `experiments/open-system-world-model/E1-DESIGN.zh.md` (SHA-256 `f06879ca…`, 2026-10-21T15:51Z). [Translator's note: Its English version is E1-design.en.md in this supplement.]

Confirmatory analysis command: `python -m osw.analyze runs/confirm runs/dev2`.

## 8. Run-time records (new in v1.1; fields are only added, training is not changed)

- Each training run records:
  - the initial validation NLL;
  - the gradient norm (before clipping) every 200 steps;
  - the parameter checkpoint after training (`runs/<phase>/ckpt/`);
  - the SHA-256 of this file at the start of the run (`prereg_v11_sha256`).
- **Check:** a recorded development job (simple τ=20, GRU, seed 102) is re-run with the new code; the validation NLL, all window metrics, the per-horizon metrics and the revival index agree bit for bit. There is also the unit test `test_v11_diagnostics_do_not_change_training`.
- **Transparency disclosure:** the round-2 development gate first returned FAIL, because the queue and the run script named the HMM/BAYES result files inconsistently ([internal record]). All 80 result files already existed at that time; after the naming was fixed, the gate was re-run and returned PASS, and no model result was changed.
