# Literature-search record (kill test KT0) (English version)

Original: KT0-NOVELTY.md (Chinese, not published). SHA-256 of the original: 92e26113a8f9ac6183158e3131caa2a7e8da9c385b97fd135ac14391f1147849. The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). Its only time evidence is the private repository history: it was committed to the private repository at 2026-10-19 12:34:48 UTC (git), in the same commit as deviations D1–D3, after the v1 pre-registration was frozen (2026-10-19 12:09:22 UTC), and the current file is byte-identical to that committed version. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

# KT0 — novelty search (2026-10-19)

**Conclusion: PASS (narrowed wording).** None of the works found compares CPTP and unitary predictors in a density-matrix world model, at long horizons, with the exact Bayes predictive distribution as the reference.

But the novelty must be narrowed to the following sentence:
> On a benchmark where the Bayes-optimal solution can be computed exactly, this work measures the long-horizon calibration of open-system (CPTP) and unitary-evolution density-matrix world models, and tests whether the difference comes from finite-dimensional recurrence.

It must not be called "the first quantum/density-matrix world model", nor "the first discovery that unitary models do not forget" — the classical version was published by others long ago. [Translator's note: this search did not cover the quantum reservoir computing literature, where the need for dissipation in density-matrix dynamics, and contraction as the remedy, were established earlier (2019–2024); the technical report's §1 ¶2 and §2 give those precedents.]

## Closest precedents (abstract or full text read for each)

| Work | What it did | Difference from this work |
|---|---|---|
| UWM-JEPA, Radha & Goktas, arXiv 2605.25313 (2026-05-25), code Apache-2.0 | Density-matrix latent + learned unitary predictor (joint dimension 8×2); treats "the joint state does not forget" as a property; hidden-velocity task, 5 steps | The paper lists long-horizon retention as an open question; no CPTP control and no Bayes reference |
| QSWM, Jiang et al., arXiv 2608.05371 (2026-08-05) | "Density-matrix-like" latent: rank 1, real matrix, trace-normalised after conjugation by a single A (not a CPTP channel); tested only on elementary cellular automata | The authors themselves report "limitations in long-horizon rollouts and in the density-matrix variant"; no unitary vs CPTP comparison, and no calibration metric |
| Gated Orthogonal Recurrent Units, Jing, Gulcehre, Peurifoy, Shen, Tegmark, Soljačić, Bengio, *Neural Computation* 31(4), 2019 (arXiv 1706.02761) | Points out that "orthogonal/unitary RNNs lack the ability to forget", and adds gating to fix this | **The classical version of the same phenomenon.** Increment of this work: belief density matrix, calibration with a Bayes reference, attribution to finite-dimensional recurrence |
| LI-QiLM, Yan, Lai, Wang, NAACL 2024 | Evolves the latent state with the Lindblad master equation in a quantum-inspired language model, for sentiment classification | Lindblad latent evolution already has a precedent in machine learning; but that is a classification task, not a world model, and it does no predictive calibration |
| Branch-JEPA (arXiv 2607.05238), MDN-VJEPA (arXiv 2610.13956) | Probabilistic JEPA world models that cite UWM-JEPA | The latent state is classical, not a density matrix |
| Kraus-Parameterized Belief RL (arXiv 2608.15715) | Uses Kraus operators to update beliefs about physical quantum systems | Physical quantum control, not a learned world model |
| Beuria & Shukla, arXiv 2608.04028 | Lindblad-inspired classical reservoir, "rotation + dissipation"; dissipation gives controllable forgetting | Classical vector state |
| HQMM (Monras 2010; Srinivasan et al. 2018; Adhikary et al. 2020) | CPTP (Kraus) sequence models | No actions, no deep encoder, no long-horizon calibration analysis |

[Translator's note: Wording in quotation marks in this table is rendered from the Chinese record. It is not the authors' verbatim text. The technical report's citations of these works were later checked against the sources; its §1–§2 give the current account.]

[Translator's note: Deviation D1 in DEVIATIONS.md (dated 2026-10-19, before any run, and committed together with this file) records that the UWM-JEPA repository (commit `1ef5735`, Apache-2.0) contains only aggregated result JSON and plotting scripts and no training code; the unitary baseline was therefore implemented from the paper's specification.]

## Impact on the experiment

- **The pre-registration is not changed.** The findings above do not affect the definitions or thresholds of H1–H5.
- **Optional exploratory baseline:** GORU (gated orthogonal RNN) as a control for "adding forgetting in the classical world". If it is added, it is recorded in DEVIATIONS.md under pre-registration §5 and labelled exploratory.
- **The prior-art table of the planning note** has been updated accordingly with three rows: QSWM, GORU and LI-QiLM.

[Translator's note: The planning note is the direction card (DIRECTION-CARD.md). The technical report and the other documents of this supplement use this name, and it is translated in excerpt as `planning-note-excerpt.en.md`.]

[Translator's note: The GORU baseline was not added. DEVIATIONS.md contains no entry for it, the experiment code contains no GORU model, and the technical report cites GORU only as prior work (§1, §2, §5).]

## Search log

2026-10-19, five groups of web-search queries in total:
1. non-unitary / dissipative / CPTP JEPA world models;
2. follow-up work citing UWM-JEPA;
3. gated orthogonal RNNs;
4. quantum-inspired belief world models + Lindblad + reinforcement learning;
5. HQMM + long horizon + calibration.

Full-text checks: QSWM pages 1–3 and the DensityQSWM transition formula (`ρ̃ = AρAᵀ` followed by trace normalisation); LI-QiLM page 1; the HTML version of UWM-JEPA (dimensions, limitations, code).

Limitations: only arXiv and the web were searched; Google Scholar and complete conference proceedings were not checked; not finding something does not mean it does not exist.
