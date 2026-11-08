# E1 design: exploratory comparison on real ETH/UCY pedestrian data (DRAFT, frozen separately before running) (English version)

Original: E1-DESIGN.zh.md (Chinese, not published). SHA-256 of the original: f06879caf9a0559799c660de13b76f4195fb7e0b37a9c1b6e826408c3c64d0b0. The original is hash-committed: this hash carries RFC 3161 tokens from DigiCert (2026-10-21 15:51:15 UTC) and Sectigo (2026-10-21 15:51:16 UTC) (`repro/docs/timestamps/2026-10-21-e1-design/`). This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

---

**Nature:** Exploratory. §2 E1 of the v1 pre-registration says only "the future direction of travel is discretised into 8 sectors; LOSO follows [internal record]" (the internal record is an earlier ETH/UCY pedestrian-trajectory experiment in the repository). This design fills in the concrete procedure and, as the development-gate diagnosis recommended, is **frozen separately**: hashed and timestamped, before any E1 training. No E1 result enters the adjudication of H1–H5, and none is made into an external claim.
**When it runs:** After the v1.1 confirmatory phase has finished, so as not to compete for CPU and affect the timing of the confirmatory runs.

[Translator's note: Despite "DRAFT" in the title, this text is the frozen version: its SHA-256 carries the RFC 3161 tokens of 2026-10-21 15:51 UTC given in the header, and the repository history records the first E1 code after that time (technical report §3.7).]

[Translator's note: A later decision publishes the E1 results in the technical report (§3.7, §4.2), where they are descriptive only and enter no hypothesis.]

## 1. Data feasibility check (2026-10-21, read-only, no training)

Raw obsmat files, 0.4 s per step; a trajectory needs at least 21 points (1 reference point + 8 context steps + 12 prediction steps).

| Scene | Trajectories long enough | Windows (stride 1) | Share of stationary steps (displacement < 0.05 m) |
|---|---:|---:|---:|
| eth | 263 | 2,343 | 5.7% |
| hotel | 99 | 1,075 | 22.7% |
| univ | 359 | 13,680 | 21.9% |
| zara01 | 138 | 2,094 | 3.2% |
| zara02 | 185 | 5,554 | 31.7% |

Stationary steps make up as much as 32%, so stationary steps cannot be forced into a heading sector.

## 2. Observation symbols (deviation from v1: O = 9)

- For each step take the displacement Δ. If |Δ| < 0.05 m, record symbol 8 (stationary).
- Otherwise take the turning angle of Δ relative to the **reference direction**, bin it into 8 sectors of 45°, and record symbols 0–7; sector 0 is centred on 0° (straight ahead).
- **Reference direction:** the direction of the last non-stationary step in the context. The whole window (context and future) is rotated by this direction, so the model sees "relative turning"; the absolute orientation of the scene does not enter the model.
- Windows whose context consists entirely of stationary steps have no reference direction and are all dropped; the number dropped will be reported.

## 3. Windows and splits

- **Windows:** 8-step context; prediction for k = 1…12 steps.
  - The training loss uses only k ∈ [1,6]; evaluation covers k ∈ [1,12]; the extrapolation window is [7,12]. This corresponds to the "short training, long evaluation" design of T1.

  [Translator's note: In the implementation this training window applied only to the gradient-trained models. The HMM (EM) and the two references of §4 (MARG and MARKOV1 in the technical report) were fitted or estimated on whole training sequences, the context plus all 12 future steps, so they also saw steps 7–12 of the training sequences (not of the test sequences) (DEVIATIONS.md D14(d); technical report §3.7, §4.2).]
- **LOSO:** 5 folds; each fold holds out 1 scene as the test set and uses all its windows (stride 1).
- The remaining 4 scenes are split into training/validation **by pedestrian** (90/10); the same pedestrian never appears on both sides.
- 3 training seeds per fold, 15 runs per model in total.

## 4. Models

- **Compared:** U2, L, K, C, GRU, LRU, HMM (8 states, EM).

  [Translator's note: In the implementation the HMM was fitted only on the first 5,000 training windows, not on all of them; because the training windows are concatenated scene by scene, this subset omits whole scenes in four of the five folds (DEVIATIONS.md D14(f); technical report §3.7).]
- **Two references:**
  - the marginal distribution on the training set, computed for each k;
  - first-order empirical transition: using only the last symbol of the context, rolled forward k steps.
- Density-matrix models: d_S = 8, with a measurement instrument of 9 outcomes; parameter counts are recomputed and reported, and aligned to within ±10% as far as possible.

  [Translator's note: The parameter counts were not matched to within ±10%: U2, L and K have 1,664 parameters, C 1,409, GRU 1,404 and LRU 1,289 (technical report §3.7).]
- Training settings as in v1.1: 6,000 steps, learning rate {3e-3, 1e-2}, selected by NLL on each fold's validation set.

## 5. Metrics and analysis (description only, no adjudication)

- **Primary metric:** mean NLL over the extrapolation window [7,12].
- **Auxiliary metrics:**
  - NLL over the training range [1,6];
  - Brier score;
  - ECE (10 bins);
  - RPS computed only on "moving" targets: the sectors are ordered by relative turning angle −135°…180° and renormalised over the 8 heading sectors. This is the counterpart, for categorical headings, of the "CRPS" in v1.
- **Comparisons:**
  - Paired difference of each model against L: computed first over the 15 (fold, seed) pairs, then a bootstrap interval over the 5 folds.
  - Only effect sizes and intervals are reported; no hypothesis tests are done and no "better than" statements are made.
- **Revival diagnostic:** roll out to k = 60 and compute the local peaks of D(k). Descriptive only; real data have no Bayes reference.

## 6. Changes needed before implementation

- The current model code hard-codes the context length as `tasks.L_CTX = 32`; it must become a passed-in parameter. This is a semantics-preserving refactor, protected by tests, with a bitwise regression check on T1.
- Add `src/osw/tasks_ethucy.py`: it reads `data/ethucy/raw/*_obsmat.txt` and performs the symbolisation, windowing and LOSO split, with unit tests covering rotation invariance, the stationary test, and no leakage in the by-pedestrian split.

## 7. Freeze

Once this design is final, timestamps for this file are obtained with a timestamping script [omitted: private file path] and stored in `docs/timestamps/`; only then is the training code written.
