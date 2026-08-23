# ESAS — Reproducibility Package

Analysis code and results for **ESAS: A Deep Learning-Based Egocentric Environmental Sound Alerting System for Deaf and Hard-of-Hearing Users on Meta Project Aria Gen 2 Glasses**. Every table in the manuscript is reproducible from the notebooks below.

## Requirements

Python 3.11.

```
pip install numpy scipy pandas scikit-learn librosa panns-inference projectaria-tools
pip install transformers torchaudio timm laion-clap   # only for notebook 06
```

## Data setup (not included in this repository)

Set `BASE` at the top of each notebook to a folder containing:

- `ESC-50/` — clone of https://github.com/karolpiczak/ESC-50 (detection training/evaluation)
- `recordings/` — 129 Aria Gen 2 VRS recordings (120 experiment trials + 9 pilots) and their MPS output folders (`mps_*_vrs/`). Not publicly distributed due to hardware-specific acquisition workflows; available from the corresponding author for academic research purposes (see the paper's Data Availability statement).
- `panns_data/` — PANNs CNN14 checkpoint (downloaded automatically by notebook 01)

The four stimulus sounds played during the acoustic evaluation (fire alarm, telephone ring, crying baby, car horn) were publicly available recordings obtained from online sources.

## Run order

`01 → 02 → 07 → 08 → 09`, plus `06` for the model comparison (~2–3 h on first run; cached afterwards). Notebook `05` is a standalone tool for inspecting the direction estimate of any single VRS recording.

## Notebook → paper mapping

| Notebook | Reproduces |
|---|---|
| `02_finetune.ipynb` | Table 2 — detection results (5-fold CV on ESC-50) |
| `06_model_comparison.ipynb` | Table 4 — AST/CLAP comparison (baseline read from `results/zeroshot_baseline.json`) |
| `07_verify_and_fix.ipynb` | Table 3 and Sect. 3.7 statistics (bootstrap CI, zero-shot baseline, Wilcoxon test); Table 5 — localization over all 120 trials (per-trial CSVs) |
| `08_mps_validation_fixed.ipynb` | MPS SLAM validation reported in the Table 5 caption and Sect. 3.6 |
| `09_latency.ipynb` | Table 6 — measured pipeline latency |

## Results files

| File | Contents |
|---|---|
| `esc50_results.json` | Per-tier detection metrics (Table 2) |
| `cv_predictions.npz` | Pooled cross-validation predictions and per-fold F1 |
| `zeroshot_baseline.json` | Zero-shot baseline per tier/fold + Wilcoxon test (Table 3, Sect. 3.7) |
| `phase1_trials.csv`, `studio_trials.csv` | Per-trial localization results for all 120 recordings |
| `localization_stats.json` | Per-angle and overall MAE / within-15° (Table 5) |
| `recording_manifest.json` | 120/120 design coverage; the 9 excluded pilot recordings |
| `mps_results.csv`, `mps_summary.json` | Per-session MPS SLAM validation (notebook 08) |
| `latency.json` | Component latency measurements (Table 6) |
| `model_comparison.json` | Macro F1 of baseline, ESAS, AST, and CLAP (Table 4) |
| `final_numbers.json` | Consolidated summary of every manuscript statistic |

## Notes

- Two sessions (`ophon90_2`, `studio_fire-90_2`) have no MPS trajectory output; their audio is analyzed normally for localization. MPS validation therefore covers 118 of 120 sessions, all of which met quality thresholds.
- The 9 pilot recordings (`Fire0*`, `Horn0*`, `Ofire0_4`) are excluded from all analyses and listed in `recording_manifest.json`.
- Earlier notebooks 03 and 04 are superseded by 07 and 08, which analyze the complete 120-trial dataset.

## License

MIT — see LICENSE.
