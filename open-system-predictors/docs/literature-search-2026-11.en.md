# Later literature searches (English record)

Written 2026-11-05, after all results in the technical report were known. Not hash-committed and carries no independent timestamp. It extends `docs/literature-search-record.en.md` (searches of 2026-10-19), which did not cover quantum reservoir computing. Searches were run by the author's review agents; they used arXiv, the open web and Semantic Scholar, not Google Scholar or complete proceedings, so absence of a hit is not proof of absence.

## 2026-10-31: quantum reservoir computing and neighbouring vocabularies

Two independent live searches, reconciled by a third reader. Vocabularies: quantum reservoir computing (fading memory, echo-state property, dissipation, contractivity, unital and non-unital channels); quantum extreme learning; quantum recurrent networks; open-system or Lindblad sequence models; hidden quantum Markov models and long-horizon prediction; quantum filtering.

Closest works read: Fujii & Nakajima 2017; Chen & Nurdin 2019; Chen, Nurdin & Yamamoto 2020; Martínez-Peña & Ortega 2023; Sannia et al. 2024; Morguí-Sancho et al. 2026 (preprint); Burgarth et al. 2013; van Handel 2009. Result: the need for dissipation and contraction as the remedy are established for input-driven reservoirs (technical report, §1 ¶2 and §2). No trained unitary-versus-CPTP long-horizon comparison against an exact Bayes or HQMM reference was found.

## 2026-11-05: searches against the gap and novelty statements

| Search | Source | Query | Result |
|---|---|---|---|
| S1 | arXiv API | "density matrix" AND "world model" | UWM-JEPA and QSWM only |
| S2 | arXiv API | "world model" AND (Lindblad, CPTP, Kraus or open quantum) | nothing relevant |
| S3 | arXiv API | "hidden quantum Markov" | 22 titles reviewed; no unitary-versus-CPTP measurement against exact Bayes |
| S4 | arXiv API | unitary + CPTP + predictor, forecast or rollout; Bayes + density matrix + unitary + sequence; "belief state" + density matrix; "quantum recurrent" + unitary + dissipative; unitary + JEPA | zero, zero, arXiv:2608.15715 only, zero, UWM-JEPA only |
| S5 | Semantic Scholar | papers citing UWM-JEPA; papers citing Chen, Nurdin & Yamamoto 2020 | Branch-JEPA and a survey (arXiv:2608.23070); 2025–26 work with fixed reservoirs |
| S6 | open web | CPTP, unitary and Bayes combinations; learned Lindblad JEPA; closed-loop quantum reservoirs; HQMM moment warm starts | none meets all three parts of the gap statement |
| S7 | arXiv API | almost-periodic or Poincaré recurrence with unitary recurrent or reservoir models | zero |

These searches also found Singh & Bhatia, arXiv:2603.05468 (recursive Kraus updates against ground-truth conditional density matrices, observations on, no unitary arm), now cited in §2. They led to the correction of two statements about cited works: that no reservoir work rolls out without observations (Fujii & Nakajima and Sannia et al. do, with the readout's output fed back as the next input), and the description of MDN-VJEPA (it reports excess NLL over a generator oracle).
