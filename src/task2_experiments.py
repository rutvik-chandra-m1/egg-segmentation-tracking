"""Task 2: run the experiment log on the dev split. Eval images are never touched here.

    python src/task2_experiments.py           run every trial, results in outputs/experiments.csv
    python src/task2_experiments.py T07 T08   run only these
"""
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

from common import OUT, ROOT, image_paths, load_gt, load_split
from metrics import score
from segment import segment

with open(os.path.join(ROOT, "src", "trials.json")) as f:
    TRIALS = json.load(f)
CSV = os.path.join(OUT, "experiments.csv")
FIELDS = ["iou", "dice", "boundary_f1", "hd95", "precision", "recall", "count_ok", "pq"]


def resolve(trial_id):
    """A trial's config is its parent's config plus its own changes."""
    trial = next(t for t in TRIALS if t["id"] == trial_id)
    parent = resolve(trial["parent"]) if trial.get("parent") else {}
    return {**parent, **trial["changes"]}


def run_one(job):
    image_id, path, config = job
    start = time.time()
    labels = segment(cv2.imread(path), config)
    return image_id, score(labels, load_gt(image_id)), time.time() - start


def evaluate(config, ids, pool):
    paths = image_paths()
    results = pool.map(run_one, [(i, paths[i], config) for i in ids])
    per_image = {i: s for i, s, _ in results}
    summary = {k: float(np.mean([s[k] for s in per_image.values()])) for k in FIELDS}
    summary["sec_per_image"] = float(np.mean([t for _, _, t in results]))
    return summary, per_image


def main():
    dev = load_split()["dev"]
    wanted = sys.argv[1:] or [t["id"] for t in TRIALS]
    rows = {}
    if os.path.exists(CSV):
        with open(CSV, newline="") as f:
            rows = {r["id"]: r for r in csv.DictReader(f)}
    with Pool(max(os.cpu_count() - 2, 1)) as pool:
        for trial in TRIALS:
            if trial["id"] not in wanted:
                continue
            summary, per_image = evaluate(resolve(trial["id"]), dev, pool)
            worst = sorted(per_image, key=lambda i: per_image[i]["iou"])[:6]
            rows[trial["id"]] = {"id": trial["id"], "name": trial["name"],
                                 **{k: f"{v:.3f}" for k, v in summary.items()},
                                 "worst": " ".join(map(str, worst))}
            print(trial["id"], trial["name"], {k: round(v, 3) for k, v in summary.items()}, "worst", worst, flush=True)
            with open(CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "name", *FIELDS, "sec_per_image", "worst"])
                writer.writeheader()
                writer.writerows(rows[t["id"]] for t in TRIALS if t["id"] in rows)


if __name__ == "__main__":
    main()
