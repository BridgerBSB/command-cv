# Retired training datasets - 2026-09-25

Deleted to free ~9.1 GB so the v3 new-park sweep could finish (it stopped at the disk floor with 10.4 GB free and ~5.7 GB still to write). Zac approved deletion on the condition that what they WERE stays documented. The images are gone; this record, the per-file listings (`<name>_files.txt.gz`), and every model trained on them remain.

**Nothing live read these folders** (grepped `scripts/ src/ tools/` for each name before deleting). Rebuilding any of them is possible from the clip CSVs + label CSVs in `output/`, subject to the video still being downloadable (sporty-clips URLs have not expired so far).

## `data/yolo_milb2`

- **Size:** 4.90 GB, 63,373 files, last modified 2026-09-18
- **Images:** train 31,548 / val 137; label files 31,685
- **Trained:** `milb10park` (weights kept in `data/yolo_ball_runs/`)
- **Why retired:** First 10-park MiLB set (built Sep 18). Built with the OLD order-dependent augmentation jitter and a smaller label set; replaced by `yolo_milb3` (incremental, order-independent build). Its only model, `milb10park`, was shelved at epoch 3/12 as a leaf and never graded. Contents are the ORIGINAL named-park clips (`aplus_*`, `aaa_*`, `aa_*`, `a_*`) plus MLB play-id clips - none of the `pull_*` rolling-loop games whose junk labels are documented in PROGRESS-LOG rows 73-74.

Train composition (clip-name prefix -> crops):

| prefix | crops |
|---|---:|
| `aplus` | 6,762 |
| `aaa` | 4,386 |
| `aa` | 351 |
| `a` | 120 |
| `6031a1a6` | 28 |
| `2944a82c` | 27 |
| `063cf4c3` | 25 |
| `6ca223aa` | 25 |
| `0145e2b2` | 24 |
| `01e1c6db` | 24 |
| `0305f843` | 24 |
| `03b6bd6b` | 24 |
| `0465f2e0` | 24 |
| `0537b901` | 24 |
| `05c82dcd` | 24 |
| `06a76e9c` | 24 |
| `09fad439` | 24 |
| `0a5b8c89` | 24 |
| `0b7d12de` | 24 |
| `0c2ae292` | 24 |

Val composition:

| prefix | crops |
|---|---:|
| `ogando` | 137 |

Full file list: `docs/retired_datasets/yolo_milb2_files.txt.gz`

## `data/yolo_ball5`

- **Size:** 3.12 GB, 36,919 files, last modified 2026-09-12
- **Images:** train 12,430 / val 6,028; label files 18,458
- **Trained:** `ball_3park_b8`, `ball_3park_gpu`, `smoke` (weights kept in `data/yolo_ball_runs/`)
- **Why retired:** MLB 3-park crop set (Camden / Detroit / Colorado) built from the POSE path (projected trajectory labels), Sep 12-15. Trained the MLB-only detectors `ball_3park_b8` / `ball_3park_gpu` (3.77-4.49 px at MLB). Superseded for MiLB work by the MiLB sets; the MLB weights it produced are KEPT in `data/yolo_ball_runs/`. `verify_ball_dataset.py` recorded its label check: contrast +21.7 at the box vs +2.0 at a control.

Train composition (clip-name prefix -> crops):

| prefix | crops |
|---|---:|
| `0145e2b2` | 24 |
| `0305f843` | 24 |
| `03b6bd6b` | 24 |
| `0465f2e0` | 24 |
| `0537b901` | 24 |
| `05c82dcd` | 24 |
| `06f18ec2` | 24 |
| `0a56b1d2` | 24 |
| `0c46dfd7` | 24 |
| `0f9856d1` | 24 |
| `10fa1c08` | 24 |
| `11825188` | 24 |
| `15393591` | 24 |
| `19aa82db` | 24 |
| `1acfc643` | 24 |
| `1d74a9a9` | 24 |
| `1ea024e4` | 24 |
| `1f02a1b1` | 24 |
| `1ff3d9e3` | 24 |
| `2247b1ab` | 24 |

Val composition:

| prefix | crops |
|---|---:|
| `01e1c6db` | 24 |
| `0899c69c` | 24 |
| `09a1055c` | 24 |
| `131f2381` | 24 |
| `13f8b651` | 24 |
| `16301919` | 24 |
| `1750dad8` | 24 |
| `24439e43` | 24 |

Full file list: `docs/retired_datasets/yolo_ball5_files.txt.gz`

## `data/yolo_ball`

- **Size:** 1.06 GB, 12,869 files, last modified 2026-09-12
- **Images:** train 5,164 / val 1,269; label files 6,433
- **Trained:** `ball_camden` (weights kept in `data/yolo_ball_runs/`)
- **Why retired:** The very first ball set: Camden Yards (game 824821) only, 5,540 labels / 277 clips, Sep 12. Trained `ball_camden`. It is still the DEFAULT `--out` of `build_ball_dataset.py` and default `--data` of `train_ball_detector.py`, so a run with no arguments would rebuild/expect it - every real run passes an explicit path.

Train composition (clip-name prefix -> crops):

| prefix | crops |
|---|---:|
| `0465f2e0` | 24 |
| `06022e12` | 24 |
| `09fad439` | 24 |
| `0b0cf4be` | 24 |
| `0f9856d1` | 24 |
| `1147e470` | 24 |
| `13e8f829` | 24 |
| `17331c80` | 24 |
| `1acfc643` | 24 |
| `24127368` | 24 |
| `267b56d0` | 24 |
| `28ee2733` | 24 |
| `2e3b3f02` | 24 |
| `378ad0c0` | 24 |
| `39315337` | 24 |
| `3bbec842` | 24 |
| `42ab4f7e` | 24 |
| `453c7725` | 24 |
| `56087b09` | 24 |
| `5bca0631` | 24 |

Val composition:

| prefix | crops |
|---|---:|
| `22fd8c1c` | 24 |
| `30b1b1f3` | 24 |
| `3df6fcae` | 24 |
| `41a5742a` | 24 |
| `5a706828` | 24 |
| `69bc65d2` | 24 |
| `6d5e9ab1` | 24 |
| `719ad396` | 24 |

Full file list: `docs/retired_datasets/yolo_ball_files.txt.gz`
