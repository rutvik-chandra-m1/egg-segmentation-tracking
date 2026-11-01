"""Task 3: score the frozen algorithm against the SAM masks and show where it fails.

    python src/task3_evaluate.py

Run once, after segment.FINAL is frozen. Writes tables and figures to outputs/eval/.
"""
import json
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import OUT, image_paths, load_gt, load_split
from make_split import SCENES
from metrics import score
from segment import FINAL, segment

EVAL_DIR = os.path.join(OUT, "eval")
METRICS = ["iou", "dice", "precision", "recall", "boundary_f1", "hd95", "pq", "count_ok"]


def classify(row):
    """One primary failure type per image, checked from most to least severe."""
    if row.pred_eggs == 0:
        return "no detection"
    if row.iou >= 0.9 and row.count_ok:
        return "good"
    if row.inst_fn and row.inst_fp and row.pred_eggs < row.gt_eggs:
        return "merged or missed egg"
    if row.pred_eggs < row.gt_eggs:
        return "missed egg"
    if row.pred_eggs > row.gt_eggs:
        return "extra region"
    if row.iou >= 0.9:
        return "good"
    if row.recall < 0.3 or row.precision < 0.3:
        return "wrong object"
    return "leak onto background" if row.precision < row.recall else "egg under-covered"


def error_map(image, pred, gt):
    """Green: correct. Red: false positive. Blue: false negative."""
    out = (image * 0.45).astype(np.uint8)
    out[(pred > 0) & (gt > 0)] = (70, 190, 70)
    out[(pred > 0) & (gt == 0)] = (40, 40, 240)
    out[(pred == 0) & (gt > 0)] = (240, 90, 30)
    return out


def bootstrap_ci(values, n=2000):
    rng = np.random.default_rng(0)
    means = [rng.choice(values, len(values)).mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def gallery(ids, predictions, paths, path, title):
    fig, axes = plt.subplots(len(ids), 4, figsize=(9, 2.3 * len(ids)), squeeze=False)
    for row, image_id in zip(axes, ids):
        image = cv2.imread(paths[image_id])
        gt, pred = load_gt(image_id), predictions[image_id]
        s = score(pred, gt)
        panels = [image[..., ::-1], (image * 0.4 + 150 * (gt > 0)[..., None]).clip(0, 255).astype(np.uint8)[..., ::-1],
                  (image * 0.4 + 150 * (pred > 0)[..., None]).clip(0, 255).astype(np.uint8)[..., ::-1],
                  error_map(image, pred, gt)[..., ::-1]]
        for ax, panel, name in zip(row, panels, ("image", "SAM (GT)", "classical", "errors")):
            ax.imshow(panel)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(name, fontsize=8)
        row[0].set_ylabel(f"#{image_id}\nIoU {s['iou']:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    assert FINAL, "freeze segment.FINAL first"
    os.makedirs(EVAL_DIR, exist_ok=True)
    split = load_split()
    paths = image_paths()
    scene_of = {i: n for n, v in SCENES.items() for i in v}
    rows, predictions = [], {}
    for part in ("dev", "eval"):
        for image_id in split[part]:
            image = cv2.imread(paths[image_id])
            pred = segment(image, FINAL)
            gt = load_gt(image_id)
            predictions[image_id] = pred
            rows.append({"id": image_id, "split": part, "scene": scene_of[image_id],
                         "gt_eggs": int(gt.max()), "pred_eggs": int(pred.max()),
                         "egg_area": float((gt > 0).mean()), **score(pred, gt)})
    df = pd.DataFrame(rows)
    df["failure"] = [classify(r) for r in df.itertuples()]
    df.to_csv(os.path.join(EVAL_DIR, "per_image.csv"), index=False)

    evaluation = df[df.split == "eval"]
    summary = df.groupby("split")[METRICS].mean().T
    summary["eval_ci_low"], summary["eval_ci_high"] = zip(*[bootstrap_ci(evaluation[m].values) for m in METRICS])
    summary.to_csv(os.path.join(EVAL_DIR, "summary.csv"))
    print(summary.round(3))

    tp, fp, fn = evaluation.inst_tp.sum(), evaluation.inst_fp.sum(), evaluation.inst_fn.sum()
    instance = {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)), "median_iou": float(evaluation.iou.median()),
                "images_iou_ge_0.9": int((evaluation.iou >= 0.9).sum()), "images": len(evaluation)}
    with open(os.path.join(EVAL_DIR, "instance.json"), "w") as f:
        json.dump(instance, f, indent=1)
    print(instance)

    scenes = evaluation.groupby("scene").agg(images=("id", "size"), iou=("iou", "mean"), boundary_f1=("boundary_f1", "mean"),
                                             precision=("precision", "mean"), recall=("recall", "mean"),
                                             count_ok=("count_ok", "mean")).sort_values("iou")
    scenes.to_csv(os.path.join(EVAL_DIR, "by_scene.csv"))
    print(scenes.round(3))
    failures = evaluation.failure.value_counts()
    failures.to_csv(os.path.join(EVAL_DIR, "failures.csv"))
    print(failures)
    worst = evaluation.sort_values("iou")
    worst[["id", "scene", "iou", "boundary_f1", "precision", "recall", "gt_eggs", "pred_eggs", "failure"]].round(3).to_csv(
        os.path.join(EVAL_DIR, "eval_sorted.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    order = worst.reset_index(drop=True)
    axes[0].bar(range(len(order)), order.iou, color=["#c0392b" if v < 0.9 else "#2d7dd2" for v in order.iou])
    axes[0].axhline(0.9, color="#555", lw=0.8, ls="--")
    axes[0].set(title="Eval images, sorted by IoU", xlabel="image rank", ylabel="IoU", ylim=(0, 1))
    axes[1].scatter(evaluation.recall, evaluation.precision, c="#2d7dd2", s=22)
    axes[1].plot([0, 1], [0, 1], color="#999", lw=0.8)
    axes[1].set(title="Precision vs recall", xlabel="recall (below line: leaks)", ylabel="precision (above: under-covers)",
                xlim=(0, 1.02), ylim=(0, 1.02))
    axes[2].scatter(evaluation.egg_area * 100, evaluation.iou, c="#2d7dd2", s=22)
    axes[2].set(title="IoU vs egg size", xlabel="egg area, % of image", ylabel="IoU", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "plots.png"), dpi=130)
    plt.close(fig)

    gallery(list(worst.id[:5]), predictions, paths, os.path.join(EVAL_DIR, "worst.png"), "Lowest-IoU eval images")
    gallery(list(worst.id[-3:]), predictions, paths, os.path.join(EVAL_DIR, "best.png"), "Highest-IoU eval images")


if __name__ == "__main__":
    main()
