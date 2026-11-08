# Material for the technical report (accumulated as the experiments proceed; not conclusions): excerpt (English version)

> Original: REPORT-NOTES.md (Chinese, not published). SHA-256 of the original: 5fabb4f9571d0d4597b43f2d19580bbd66fd6506452e3a4748abbfe9697075af. The original is not hash-committed: no hash of it has been sent to a timestamping service, so it carries no independent timestamp (no RFC 3161 token). Its only time evidence is the private repository history. It was first committed at 2026-10-21 17:12:14 UTC (git); the compute section translated here is unchanged since that commit. The version hashed above was committed at 2026-10-23 01:05:03 UTC (git), when sections not translated here were added. This English version was prepared on 2026-10-30, after all results reported in the technical report were known. The Chinese original is the binding text; omissions are marked [omitted: …] and translator's notes [Translator's note: …].
>
> This is an excerpt: only the section on per-forward compute, which the technical report cites in the Compute item of §3.2 (Predictors), is translated. The other sections of the original are not included.

## Compute per forward pass (reporting required by pre-registration §3; measured 2026-10-21)

- **Measurement:** `torch.utils.flop_counter.FlopCounterMode`, batch 1, a context of 32 steps plus a blind rollout of 50 steps, one action.
- **Only** matrix-multiplication-type operators are counted. The following items are **not counted**, so the numbers in the table are lower bounds:
  1. Matrix exponentials: the 64×64 superoperator `matrix_exp` of L and L1, and the (8d_E)×(8d_E) `matrix_exp` of the U family. Each is computed only once per forward pass and per action, and then shared within the batch.
  2. Element-wise operations and normalisation.
  3. The real-arithmetic expansion of complex matrix multiplications (about ×4).
- **The effect is largest for L:** a 64×64 complex matrix exponential, estimated by Padé plus squaring, needs about 20 × 64³ complex multiplications, i.e. about 4×10⁷ real flops. Counted for a single sequence, it would exceed the numbers in the table; amortised over a batch of 64, it is about 6×10⁵ per sequence.
- The technical report will give both the measured values in the table and this amortised estimate, not just one number.

  [Translator's note: The technical report does not restate the measured values or the amortised estimate. The Compute item of its §3.2 (Predictors) cites this excerpt and lists the three uncounted items above.]

| Model | Parameters | Matrix-multiplication FLOPs / forward |
|---|---:|---:|
| U1 | 640 | 471,040 |
| U2 | 1,024 | 1,828,864 |
| U4 | 2,560 | 11,905,024 |
| U8 | 8,704 | 89,368,576 |
| L | 1,024 | 971,776 |
| L1 | 768 | 969,728 |
| K | 1,024 | 977,920 |
| C | 769 | 472,064 |
| GRU | 1,099 | 160,980 |
| LRU | 1,004 | 90,080 |

[Translator's note: The parameter counts are the nominal one-action counts of the predictor table in §3.2 of the technical report; that table also gives, in brackets, the counts of parameters movable under Adam training (post hoc).]
