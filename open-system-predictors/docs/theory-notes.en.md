# Theory draft: the "early–late" calibration trade-off for finite-dimensional unitary belief latent variables (not adversarially reviewed; not for external use): working notes, not peer reviewed (English version)

Original: THEORY-DRAFT.md (Chinese, not published). SHA-256 of the original: 37e3f7bf6377f840985e77c0457883711a6195f316552330653d3ed50c6d8598. The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). Its only time evidence is the private repository history. It was first committed at 2026-10-19 12:40:15 UTC (git); the numerical checks of 2026-10-19 below are unchanged since that commit. The version hashed above was committed at 2026-10-21 15:03:02 UTC (git), when the Dirichlet bound was struck through, limitation item 2 was revised, the theorem's implications for the experiment were narrowed, and the 2026-10-21 correction, limitation item 5 and the addendum were added. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].

These are working notes, not peer reviewed. This English version translates every section of the original; the only omissions are the redactions marked in place.

[Translator's note: The original's title restricts external use. A later decision publishes this translation, labelled as working notes that have not been peer reviewed, in the supplement to the technical report.]

---

**Status:** DRAFT. [omitted: internal administrative note] Before it is used in a paper, it must first pass targeted error-hunting [omitted: internal tool name], and an independent numerical check must be completed.

## Setting

- **Unitary model:** joint space ℋ = ℋ_S ⊗ ℋ_E, finite-dimensional, D = d_S·d_E. The filtered initial state is ρ. Blind rollout ρ_k = U^k ρ U^{−k}. Readout p_k(o) = Tr[(E_o⊗I) ρ_k], where {E_o} is a POVM.
- **Target process:** a finite, primitive hidden Markov process whose Bayes k-step predictive distribution q_k converges at a geometric rate to the stationary readout q_∞: ‖q_k − q_∞‖₁ ≤ C·λ^k, 0 < λ < 1. In T1-simple, λ = e^{−1/τ}.
- **Distance:** TV(p, q) = ½‖p − q‖₁.

## Lemma 1 (almost periodicity)

Let U = Σ_j e^{−iθ_j} Π_j be the spectral decomposition. Then

p_k(o) = Σ_{j,l} e^{−i(θ_j − θ_l)k} · Tr[(E_o⊗I) Π_j ρ Π_l].

So k ↦ p_k is a (vector-valued) trigonometric polynomial in finitely many frequencies, and is a Bohr almost periodic function.

**Corollary (Bohr):** For any ε > 0, the set of ε-almost periods T_ε = { τ ∈ ℕ : sup_k ‖p_{k+τ} − p_k‖₁ ≤ ε } is relatively dense: there is an inclusion length ℓ(ε) such that every integer interval of length ℓ(ε) contains at least one element of T_ε.

~~**Upper bound on ℓ(ε):** The phase differences take at most r ≤ D² distinct frequencies. By Dirichlet's simultaneous approximation theorem, there is an upper bound that depends only on r and ε, of order (c·r/ε)^r. This is the source of "the larger the environment dimension, the longer the recurrence interval", corresponding to H2.~~

**Correction (2026-10-21, adopting D04 of `GATE-DIAGNOSIS.zh.md` [omitted: private file path]): the paragraph above is wrong.**
- Dirichlet's theorem only gives an upper bound on the time of the **first** near-return. It does not give an inclusion length such that "every window of length ℓ contains a near-period".
- An upper bound on the inclusion length that depends only on D and ε does not exist. Counterexample: D = 2, ε = 0.1, U = diag(1, e^{2πi/N}), ρ = |+⟩⟨+|, measurement {|+⟩⟨+|, complement}. Then p_k = ((1+cos(2πk/N))/2, (1−cos(2πk/N))/2). For τ ∈ [N/6, N/3], ‖p_τ − p_0‖₁ ≥ 1/2 > ε, and N can be arbitrarily large.
- For a **fixed** U, relative density (Bohr) still holds, but ℓ(ε) depends on the spectrum of U, not only on the dimension.
- So "the larger the environment dimension, the longer the recurrence interval" is not a theorem. Another counterexample: replace U by U⊗I and leave the readout unchanged; the whole prediction sequence is then exactly unchanged. **H2 is only an empirical hypothesis** and cannot be derived from this draft.

[Translator's note: GATE-DIAGNOSIS.zh.md is the development-gate diagnosis; a summary of it is included in this supplement (development-gate-diagnosis-summary.en.md). D04 here and D05 below are that diagnosis's finding labels, not the deviations D1–D16 of DEVIATIONS.md.]

## Proposition 1 (early–late trade-off)

For any k₀ ≥ 1, any K ≥ k₀ and any ε > 0, there exists, in the window [K, K + ℓ(ε)) of length ℓ(ε), some k such that

  TV(p_{k₀}, q_{k₀}) + TV(p_k, q_k) ≥ TV(q_{k₀}, q_∞) − ε/2 − ½·C·λ^K.

**Proof:**
1. Take τ ∈ T_ε ∩ [K − k₀, K − k₀ + ℓ(ε)), and let k = k₀ + τ ∈ [K, K + ℓ(ε)).
2. By almost periodicity, ‖p_k − p_{k₀}‖₁ ≤ ε, i.e. TV(p_k, p_{k₀}) ≤ ε/2.
3. By the triangle inequality:
   TV(q_{k₀}, q_∞) ≤ TV(q_{k₀}, p_{k₀}) + TV(p_{k₀}, p_k) + TV(p_k, q_k) + TV(q_k, q_∞),
   where TV(p_{k₀}, p_k) ≤ ε/2 and TV(q_k, q_∞) ≤ ½·C·λ^k ≤ ½·C·λ^K.
4. Rearranging gives the result. ∎

**Meaning:** Let k₀ be an early horizon at which the Bayes prediction is still informative, i.e. TV(q_{k₀}, q_∞) is large. If a finite-dimensional unitary predictor is accurate at k₀, then **in every window of length ℓ(ε), however late,** it must at some step make an error close to TV(q_{k₀}, q_∞). It cannot be both "accurate early" and "accurate late".

Pinsker's inequality KL ≥ 2·TV² then gives a lower bound in KL form.

## Proposition 2 (the CPTP family contains the Bayes-optimal predictor)

Let S ≤ d_S. On the state basis |s⟩, construct:
- **Channel:** Kraus operators K_{s→s'} = √P_{ss'} · |s'⟩⟨s|. Since Σ_{s,s'} K†K = Σ_s |s⟩⟨s| Σ_{s'} P_{ss'} = I, this is a CPTP map; on diagonal states it acts exactly as the Markov matrix P, and it removes the off-diagonal elements.
- **Instrument:** M_o = Σ_s √B_{so} · |s⟩⟨s|, satisfying Σ_o M_o†M_o = I.

Starting from the diagonal initial state diag(π), the filtering and k-step readout of this model are **exactly** equal to the Bayes forward algorithm, so its error is 0 at every horizon. This is the known fact that HQMMs contain HMMs, and it is also consistent with the "classical inclusion" of QSWM.

## Theorem (separation of representational capacity, draft)

For a primitive hidden Markov process of the T1 type with a non-trivial transient:
- in the CPTP family with dimension d_S ≥ S there is a predictor that agrees with Bayes at every horizon (Proposition 2);
- every finite-dimensional unitary predictor family is constrained by Proposition 1: the sum of its "early error" and its "late worst-window error" is bounded below by TV(q_{k₀}, q_∞) minus an arbitrarily small amount.

**What this means for the experiment (narrowed 2026-10-21):**
- In the extrapolation window [51,200], the error of a unitary model depends on the ℓ(ε) of that U (determined by its spectrum) relative to the window length. H2's prediction that "the larger d_E, the smaller the extrapolation error" is **only an empirical hypothesis**, not a corollary of the theorem (see the correction above).
- Proposition 1 only gives a lower bound on the error at "some step"; it does not imply a **mean** KL ≥ 0.02 over [51,200]: sparse bad points are diluted by the average. The effect size of H1 is an empirical question.
- Proposition 1 covers only blind rollout with a fixed U; when the action sequences differ, the product U(a_k)…U(a_1) is not covered.

## Known limitations and open items

1. **Novelty:** Almost periodicity of finite-dimensional unitary dynamics is a known result dating from 1957; GORU (2019) pointed out "cannot forget" for classical RNNs. The contribution of this proposition lies only in stating it as an early–late trade-off for **predictive calibration**, and in giving the exact CPTP inclusion as a contrast. This must be stated truthfully in the related work.
2. **Constants:** ~~The Dirichlet upper bound on ℓ(ε) is very loose.~~ An upper bound that depends only on the dimension does not exist (see the correction); ℓ(ε) can only be measured numerically, directly on the trained models.
5. **Scope of Proposition 2 (adopting diagnosis D05):** Proposition 2 is an existence result for the general CPTP family. It cannot be transcribed directly into "the current L (J=3) or K (rank 4) has zero error". The L family has an exact solution when J ≥ 7, see the addendum at the end; for J=3 and rank 4 there is no known exact solution. In addition, when d_S > S, the listed Kraus operators and instrument cover only the S-dimensional subspace and must be completed on the orthogonal complement to be trace-preserving on the whole space; in this experiment d_S = S = 8, so it is not affected.
   [Translator's note: The technical report (§3.2, "Representability") records that later internal reviews found exact predictors of T1-simple within the registered three-jump L family and within the rank-4 K family. This supersedes "no known exact solution" in this item; see also the note in the addendum below.]
3. **Filtering phase:** The proposition covers only blind rollout; the measurement updates of the context phase are non-unitary, which does not affect the conclusion.
4. **To do:** targeted error-hunting with [omitted: internal tool name]; a numerical check with `tests/test_core.py::test_recurrence_vs_relaxation_signature`, extended to the trained models; a check of the derivation in Lean or symbolically in small dimensions (optional).

[Translator's note: The original numbers this list 1, 2, 5, 3, 4: item 5 was inserted later. The order and numbers are kept; a Markdown renderer may display them as 1–5.]

## Numerical checks (2026-10-19, independent of training)

- **Proposition 2:** On T1-simple (τ=20), the difference between the CPTP model built by the classical embedding and the Bayes prediction is at most **2.3×10⁻¹⁵** over 50 contexts × 120 steps, i.e. exact agreement.
- **Proposition 1:** A random 16-dimensional unitary matrix was generated (Hamiltonian scale 0.3), and the number of times p_k returns near p_5 was counted over 20,000 steps:

  | ε | Returns | First return | Largest gap |
  |---|---|---|---|
  | 0.1 | 205 | k = 321 | 482 |
  | 0.05 | 22 | k = 321 | 3,755 |
  | 0.02 | 0 | — | — |

  [Translator's note: no script or seed for this illustration is preserved, and it does not describe the trained models, whose predictions return inside the window (technical report, §5, Recurrence).]

  **This is a risk for the pre-registration:** the first return occurs at k=321, already beyond the extrapolation window [51,200]. So within this window, H1's gap may come mainly from "non-converging oscillation" rather than from full recurrence, and the effect size may be small. The pre-registration is frozen and the window is not changed; if H1 does not reach its threshold, this will be reported truthfully, and long-horizon rollout (k up to 600, revival index) will be used as mechanistic evidence.

## Addendum (2026-10-21): an exact construction for Proposition 2 within the GKSL semigroup family, and the gap at J=3

Proposition 2 uses a general CPTP channel. In the experiment, L is a GKSL semigroup (exp(𝓛) per step) and K is a rank-4 Kraus channel; neither automatically contains an exact solution, so each has to be checked separately.

**Exact GKSL construction for T1-simple (J = 7):** Let T be the cyclic shift T|s⟩ = |s+1⟩, and take:
- H = i·log T (Hermitian);
- jump operators L_j = √(γ/8)·T^j, j = 1…7, γ = 1/τ.

The dissipative part commutes with Ad_T, and maps diagonal states to diagonal states and coherences to coherences. Therefore:
- on the diagonal block, exp(𝓛) = Ad_T ∘ [Π + e^{−γ}(I − Π)] = λC + (1 − λ)Π;
- coherences do not leak into the diagonal.

Numerical check: 8 basis-state inputs, maximum error **2.0×10⁻¹⁵**.

**J = 3 (the parameter-matched configuration of L in the experiment):**
- With shift jump operators only, there is no exact solution: for the real parts of all 7 nonzero eigenvalues to equal −γ, at least four classes of shift are needed: 1, 2, 3 and 4.
- The best Nelder–Mead approximation has a maximum element-wise transition-matrix error of about **0.006**. This is only an upper bound, because the same optimiser reached only 1.1×10⁻³ at J=7, where an exact solution exists.
- General (non-permutation) jump operators leak coherences into the diagonal; whether they can do better is unknown.

[Translator's note: The technical report (§3.2, "Representability") records that a later internal review found an exact predictor of T1-simple within the registered three-jump L family: H = i·log X with X the cyclic shift |s⟩ ↦ |s+1⟩, jump operators L_j = √(1/(4τ))·X^j (j = 1, 2, 3) and the diagonal instrument. The report states that this supersedes the three-jump approximation above (maximum transition-matrix error about 0.006). The eigenvalue condition in the first bullet above is a condition for reproducing the eight-state transition matrix exactly; the report's three-jump predictor reproduces the Bayes predictive distribution without reproducing that matrix (the report's §3.1 notes that the observation process of T1-simple is exactly that of a 4-state HMM, with states s and s + 4 merged).]

**Implications for the experiment:**
- The structural gap between the parameter-matched L(J=3) and the exact solution is small (element-wise error of order ≤ 0.006).
- That L stopped at an in-range KL≈0.03 in the development phase is mainly an optimisation problem, not a lower limit of representational capacity. Consistent with this: after the learning rate was changed to 1e-2, the KL fell to 0.012 (seed 101).
