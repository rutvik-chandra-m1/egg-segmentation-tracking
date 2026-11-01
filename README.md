# CV Intern Task: Egg Segmentation and Tracking

Segments eggs with SAM, builds a classical algorithm to match it, evaluates that algorithm against SAM, and tracks two eggs through a video.

**Full write-up: [report/report.pdf](report/report.pdf)**, including the experiment log and evaluation.

**Tracking video: [watch on Google Drive](https://drive.google.com/file/d/1Ne4JJu_ycORmcKbMRhbO92BvLx1tToE5/view?usp=sharing)**

| Task | Result |
|---|---|
| 1. SAM masks | All 110 images, each mask reviewed |
| 2. Classical algorithm | 40 logged trials, split by scene to limit leakage |
| 3. Evaluation, eval split | IoU 0.873 (95% CI 0.76 to 0.96), median 0.984 |
| 4. Tracking | 2 IDs, 0 wrong identities on 64 hand-labelled checks |

The eval score is flattered by an easy scene mix; the dev IoU of 0.804 is the safer guide. The report explains why.

## Setup

Tested on Windows with Python 3.14 on CPU. No GPU is needed.

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux or macOS, use `.venv/bin/python` instead.

### Data and weights

```bash
mkdir -p data/raw weights
```

```bash
curl -L -o data/egg.v1i.folder.zip "https://drive.usercontent.google.com/download?id=1a58rACcCFnaj0ZiDoNyGbDWXunRyWziv&export=download&confirm=t"
```

```bash
unzip -q data/egg.v1i.folder.zip -d data/raw
```

```bash
curl -L -o weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

```bash
yt-dlp -f "best[height<=360]" -o data/video.mp4 "https://www.youtube.com/watch?v=Y4JyV94y2_A"
```

## Running

SAM masks and the split are committed, so steps 2 to 4 work without running step 1.

| Step | Command | Time on 8 CPU cores |
|---|---|---|
| 1a. SAM candidates | `python src/task1_sam.py generate` | 45 min |
| 1b. Select masks, apply reviewed prompts | `python src/task1_sam.py select` | 20 min |
| 2a. Dev/eval split | `python src/make_split.py` | 1 min |
| 2b. Experiment log, dev split only | `python src/task2_experiments.py` | 15 min |
| 3. Evaluate the frozen algorithm | `python src/task3_evaluate.py` | 5 min |
| 4. Track the video | `python src/task4_track.py` | 4 min |
| Report | `python src/build_report.py --github <url> --video <url>` | seconds |

Metrics have a self-check: `python src/metrics.py`.

To try the segmenter on one image:

```python
import cv2
from segment import FINAL, segment

labels = segment(cv2.imread("egg.jpg"), FINAL)  # 0 is background, 1..n are eggs
```

`FINAL` expects images around 640 px wide, as in the dataset.

## Layout

| Path | Contents |
|---|---|
| `src/task1_sam.py` | SAM mask generation and selection |
| `src/task1_prompts.json` | Reviewed point and box prompts for 58 images |
| `src/make_split.py` | Scene groups and the leakage check |
| `src/segment.py` | Classical algorithm; `FINAL` is the frozen config |
| `src/trials.json` | Every trial: what changed, why, and its parameters |
| `src/task2_experiments.py` | Runs trials on the dev split |
| `src/metrics.py` | IoU, Dice, boundary F1, HD95, panoptic quality |
| `src/task3_evaluate.py` | Eval metrics, failure analysis and figures |
| `src/task4_track.py` | Tracker |
| `src/video_keyframes.json` | Hand-labelled egg positions used to score identity |
| `splits/split.json` | Dev and eval image IDs |
| `outputs/` | SAM masks, experiment CSV, evaluation results |

## Dataset

[egg, Roboflow Universe](https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/1), CC BY 4.0. Roboflow's train/valid/test folders are ignored; see the report for why.

## Known quirk

GrabCut is randomly initialised, so scores can differ by about 0.001 between runs.
