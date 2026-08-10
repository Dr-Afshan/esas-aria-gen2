# ESAS — Egocentric Sound Alerting System

Real-time environmental sound detection and direction estimation for deaf and hard-of-hearing (DHH) users on Meta Project Aria Gen 2 AR glasses.

---

## Results

| Metric | Value |
|---|---|
| Detection Macro F1 | 0.921 |
| Frontal MAE | 5.5 degrees |
| Overall MAE | 57.6 degrees |
| System latency | ~480 ms |
| AST (comparison) | F1=0.951 |
| CLAP (comparison) | F1=0.955 |

---

## Setup

```bash
git clone https://github.com/Dr-Afshan/esas-aria-gen2.git
cd esas-aria-gen2
python3 -m venv esas_env
source esas_env/bin/activate
pip install -r requirements.txt
```

Set `BASE` in each notebook to your local project path.

---

## Notebooks

Run in order:

| Notebook | Description | Requires |
|---|---|---|
| 01_download_models | Download PANNs checkpoint | Internet |
| 02_finetune | Fine-tune on ESC-50, F1=0.921 | ESC-50 dataset |
| 03_statistics | Bootstrap CI and Wilcoxon test | Notebook 02 |
| 04_mps_validation | MPS SLAM validation | VRS recordings |
| 05_direction_from_vrs | GCC-PHAT localization, MAE=5.5 deg | VRS recordings |
| 06_model_comparison | AST and CLAP comparison | Notebook 02 + transformers |

Notebooks 01, 02, 03, and 06 run on the public ESC-50 dataset.
Notebooks 04 and 05 require Aria Gen 2 VRS recordings (available on request).

---

## Structure

```
esas/
  detection.py          PANNs CNN14 sound detection
  localization.py       GCC-PHAT direction estimation
  sound_alert_system.py end-to-end pipeline

demo/
  esas_live.py          live demo (--demo or --vrs modes)
  aria_direction.py     direction from a VRS file
  extract_audio.py      export VRS audio to WAV
```

---

## Demo

```bash
# Demo mode — no hardware needed
python demo/esas_live.py --demo

# Replay a recording
python demo/esas_live.py --vrs /path/to/recording.vrs
```

Note: Live streaming from Aria Gen 2 is not supported on macOS with SDK 2.2.0.

---

## Data

Recordings are in Aria Gen 2 VRS format and available on request.
ESC-50: https://github.com/karoldvl/ESC-50

---

## Citation

```bibtex
@article{hashmi2026esas,
  title  = {{ESAS}: An Egocentric Environmental Sound Alerting System
             for Deaf and Hard-of-Hearing Users Using Wearable {AR} Glasses},
  author = {Hashmi, Afshan},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT — see LICENSE.
