# Omytea research

Pre-registered experiments from Omytea's research: for each study, the code, tests, English versions of the pre-registrations and designs, the report and timestamp proofs. Run records, model checkpoints and data files are in each study's archived Zenodo package, linked below. The binding design documents are not published; their timestamped hashes and English versions are.

Omytea is building a quantum-information world-model protocol in which streaming vector memory is compiled into density states, evolved by valid operator/channel families, measured into forecast probabilities, and revised by an outcome-scored ledger. (This is a design goal, not a result.)

So far, none of our comparisons has shown a density-matrix model forecasting better than a suitable classical model on real data. Two newer synthetic studies, unpublished and listed on our research page, also found no edge: in one, a classical hidden Markov model did better on every seed; in the other, small density-matrix models trained without a warm start and from a single start were unreliable on synthetic clocks. This describes the runs we did, not a proof that no such case exists. On real pedestrian headings the best of four density-matrix models had the lowest mean far-horizon log loss (1.149, against 1.157 for a hidden Markov model and 1.170 for a recurrent network; the other three were at 1.170–1.180), with model sizes not matched; the report rates this comparison inconclusive. The full record of our experiments, including those not published here, what went wrong, and plans, is at [omyteaai.com/research](https://omyteaai.com/research/).

## Studies

### `open-system-predictors/`: unitary versus open-system density-matrix predictors

> **Preprint, not peer reviewed.** Jiaxuan Chen (Omytea), *Open-system versus unitary density-matrix predictors on long-horizon predictive error: a pre-registered comparison against an exact Bayes reference*, Zenodo, 7 November 2026, [doi:10.5281/zenodo.24050453](https://doi.org/10.5281/zenodo.24050453). All review so far was internal and AI-assisted. None of it is external peer review.

**Question.** A *density matrix* is a small table of numbers: its diagonal holds the probabilities of the possible states, and its other entries record how those possibilities can interfere. A *unitary* density-matrix model evolves it reversibly, so it cannot dissipate the uncertainty it represents; an *open-system* model can forget, as a system that leaks into its surroundings does. On synthetic processes whose best possible forecast is known exactly, does the unitary model build up extra error far beyond the horizon it was trained on, and if so, is that because it lacks forgetting? We then ran the same model families on real pedestrian data (model sizes not matched), compared standard hidden quantum Markov models with classical sequence models at equal parameter budgets on synthetic and real streams, and tested a warm start for a training failure we found.

**What we found** (measured on our own runs; every number is in the report with its source)

- On the simple task (relaxation time 20) the unitary model carried 0.0237 nats of extra far-horizon error compared with the open-system model (95% interval 0.0217–0.0260), meeting the pre-registered bar under the registered learning-rate selection; on one development seed the unitary model's other learning rate gave 0.014, below the bar. This holds for a blind rollout that ignores the measurement; applying the measurement channel at each step removed nearly all of the extra error (post hoc), so the result depends on the rollout convention. On the two-timescale task with the same slow relaxation time, the gap was 0.0189, below the 0.02 bar.
- Contraction alone removed that extra error (pre-registered kill test KT3). This is a finding about the relaxation-time-20 tasks, whose far horizon holds almost no predictable signal; where signal remained (relaxation time 80), the trained contracting models captured it unevenly. On the two-timescale task with a slow relaxation time of 80, plain contraction did slightly better than the parameter-matched Lindblad model (−0.0022 nats, 95% interval −0.0033 to −0.0011), so the general Lindblad generator added nothing beyond contraction there; the far window holds a single relaxation rate, so a two-timescale mechanism could not be tested. The supported claim is that forgetting helps, not that recurrence hurts. The registered dose-response test was not met.
- On real pedestrian walking directions, six of the seven learned models were within 0.031 nats of each other and most intervals include zero; the seventh, a linear recurrent unit, did worse.
- At equal parameter budgets, no hidden quantum Markov model met the registered rule against classical networks. The positive control failed, so that registered conclusion is not accepted. Descriptive only: on the real streams (three seeds) these models were not distinguishable from classical networks on weekly temperature categories and behind on household appliance energy use and taxi demand; on synthetic clocks (five seeds) they were ahead by 0.070 on a two-state clock at 64 parameters, below the registered 0.10 margin, and far behind on a three-level weakly measured clock.
- Given the true model order, a statistics-based warm start fixed a training failure on a three-level weakly measured clock (10 of 10 seeds), with the gain attributed to the warm start relative to the registered random-start recipes (one of them compute-matched). The warm start was developed on exploratory seeds of the same process, and some unregistered runs started near the identity reached the target without it. It did not transfer to a weaker-signal version (0 of 10) or to four-level development runs.

The report states what is not new: the limitation and the remedy both have prior work, cited there.

**What is in the folder.** Every file is copied unchanged from the supplement in the Zenodo record. The folder's `README.md` and `repro/data/README.md` describe the full supplement; the run records and data files they mention are not in this repository.

- `docs/`: English versions of the pre-registrations and designs, deviations, predictions and records.
- `repro/experiments/open-system-world-model/`: code (`src/`, `tests/`, `scripts/`, launchers) and the report with its analysis scripts (`report/`). Read the report at [`report/REPORT.md`](open-system-predictors/repro/experiments/open-system-world-model/report/REPORT.md), or as a PDF in the Zenodo record.
- `repro/frozen-v1.1/`: three files in the form they had when the re-registration (v1.1) was frozen. With them, every code entry of the v1.1 code manifest can be checked here; its 84 run-record entries need the Zenodo package. The code manifest of the warm-start follow-up (E3) can be checked in full here.
- `repro/docs/timestamps/`: RFC 3161 tokens from two authorities and OpenTimestamps proofs.
- `repro/data/`: the data README and a fetch script.
- `SHA256SUMS`: a hash for every file of the full supplement.

**Only in the Zenodo package.** The run records (`repro/experiments/open-system-world-model/runs/`, 2,322 files including logs and model checkpoints) and the data files: the UCI and Numenta files and the category file derived from ERA5T data. Download the supplement from the [Zenodo record](https://doi.org/10.5281/zenodo.24050453) to recompute results or re-run experiments.

## How to check it

1. **Integrity.** `cd open-system-predictors && shasum -a 256 -c --ignore-missing SHA256SUMS` (on Linux, `sha256sum -c --ignore-missing SHA256SUMS`) checks every file here against the full list; expect 140 lines ending in `OK`. `SHA256SUMS` is the same file as in the Zenodo package (SHA-256 `5cbca4f80a4b2fbb5262221f2eda69745464d1343a0599bca6e5fffb9ac6c23e`). In the unpacked Zenodo package, run the command without `--ignore-missing`.
2. **Freeze before results.** `openssl ts -reply -in <token>.tsr -text` prints the token time and the hash it covers; for a design document it must equal the hash in the supplement README table, and for the two code manifests the SHA-256 of the manifest file itself. `openssl ts -verify -digest <sha256> -in <token>.tsr -CAfile <authority chain>` checks the signature. A token shows the design existed no later than the token time; the run logs in the Zenodo package place each governed run after it.
3. **Recompute.** Result numbers in the running text are listed with their sources in `report/numbers.json`; tables are rendered from the run records in the Zenodo package; a few values come from unpublished records, as the report states.
4. **Re-run.** Environment used: Python 3.11.16, PyTorch 2.11.0, NumPy 2.4.4, SciPy 1.17.1, pandas 3.0.5, CPU only. Tests: in `open-system-predictors/repro/experiments/open-system-world-model/`, create an empty `E3-DESIGN.zh.md` (the warm-start code records the hash of its design file; see `open-system-predictors/README.md`, section 2, "Tests"; best done in a copy, since the run writes files), then run `PYTHONPATH=src python3 -m pytest tests`. Expected without the Zenodo data: 166 passed, 11 skipped, 1 failed; the failing test needs `runs/e2`, and the skipped ones need the data files, the ETH/UCY download, or `E3_SLOW=1` for one slow test. Results that depend on eigendecompositions reproduce exactly only on the same OS and LAPACK version; the supplement README lists the differences we measured after an OS change.

## Data and attribution

- Synthetic data: generated by the code from fixed seeds.
- ETH and UCY pedestrian recordings (Pellegrini et al., ICCV 2009; Lerner et al., 2007): not redistributed; `repro/data/fetch_ethucy.sh` downloads them and checks hashes.
- UCI Appliances energy prediction (Candanedo, 2017, CC BY 4.0), the Numenta Anomaly Benchmark taxi series (MIT) and the weekly temperature category file derived from ERA5T data as provided by ECMWF's AI Weather Quest: not in this repository; they are in the Zenodo package with their notices. Attribution and licence basis: `repro/data/README.md`.
- The unitary predictor is implemented from the text of arXiv 2605.25313; no code from that project is included.

## Other public work

- [omyteaai.com/research](https://omyteaai.com/research/): all our experiments, what went wrong, and plans with their stop conditions.
- [omytea-protocol](https://github.com/Omytea-Almighty/omytea-protocol): receipt schema, scoring code and a verifier for machine predictions, with a [reproduction of a public competition leaderboard](https://github.com/Omytea-Almighty/omytea-protocol/tree/fb164030da17e2aea8d547fcdc9df524c5f7081f/examples/aiwq-reproduction) from published files.
- [omytea-quantum-substrate](https://github.com/Omytea-Almighty/omytea-quantum-substrate): a small density-matrix library.

## Licence

Copyright 2026 Jiaxuan Chen. Code: Apache License 2.0 (`LICENSE`; in the study folder, `LICENSE-CODE`). Reports, documents and figures: CC BY 4.0, as listed in the study folder's `LICENSE-TEXT-AND-DATA.md`. This README is also licensed under CC BY 4.0. Timestamp requests, tokens and proofs are included unchanged as evidence and are under neither licence. Third-party data keep their own licences.

## Cite

Use "Cite this repository" on GitHub (from `CITATION.cff`), or:

> Chen, J. (2026). *Open-system versus unitary density-matrix predictors on long-horizon predictive error: a pre-registered comparison against an exact Bayes reference* [Preprint]. Zenodo. https://doi.org/10.5281/zenodo.24050453

## What this repository does not claim

- No forecasting advantage for density-matrix models on real or synthetic data.
- No parameter-efficiency result.
- No external peer review.
- No product or customer result.
