"""Task 1: egg segmentation with SAM.

SAM is class-agnostic, so this runs in two steps:
  generate  automatic mask generator on every image, all candidates cached to disk
  select    pick the egg-like candidates, then apply hand-reviewed point prompts
            from task1_prompts.json to the images where auto-selection was wrong

    python src/task1_sam.py generate
    python src/task1_sam.py select [image ids]
"""
import json
import os
import sys
import time

import cv2
import numpy as np

from common import OUT, ROOT, SAM_MASKS, image_paths, overlay

CANDIDATES = os.path.join(OUT, "sam_candidates")
CHECKPOINT = os.path.join(ROOT, "weights", "sam_vit_b_01ec64.pth")
PROMPTS = os.path.join(ROOT, "src", "task1_prompts.json")


def load_sam():
    from segment_anything import sam_model_registry
    return sam_model_registry["vit_b"](checkpoint=CHECKPOINT)


def generate():
    from segment_anything import SamAutomaticMaskGenerator
    os.makedirs(CANDIDATES, exist_ok=True)
    amg = SamAutomaticMaskGenerator(
        load_sam(), points_per_side=10, points_per_batch=32,  # 64 px grid; smallest egg is >100 px
        pred_iou_thresh=0.8, stability_score_thresh=0.85)
    for image_id, path in image_paths().items():
        out = os.path.join(CANDIDATES, f"{image_id}.npz")
        if os.path.exists(out):
            continue
        t = time.time()
        results = amg.generate(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
        np.savez_compressed(
            out,
            masks=np.array([r["segmentation"] for r in results], bool),
            iou=np.array([r["predicted_iou"] for r in results]),
            stability=np.array([r["stability_score"] for r in results]))
        print(image_id, len(results), f"{time.time() - t:.1f}s", flush=True)


def fill_holes(mask):
    """Glints and speckles leave holes in SAM masks; an egg has none."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    return filled.astype(bool)


def egg_score(mask, image_hsv):
    """Score in [0, 1] for how much a candidate looks like an egg, 0 if ruled out."""
    area = mask.sum()
    if not 0.004 < area / mask.size < 0.6:
        return 0.0
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 20:
        return 0.0
    (cx, cy), (d1, d2), angle = cv2.fitEllipse(contour)
    if not 1.0 <= max(d1, d2) / max(min(d1, d2), 1) <= 2.2:  # 2.2: dataset is stretched to square
        return 0.0
    ellipse = np.zeros(mask.shape, np.uint8)
    cv2.ellipse(ellipse, ((cx, cy), (d1, d2), angle), 1, -1)
    ellipse_iou = (mask & (ellipse > 0)).sum() / max((mask | (ellipse > 0)).sum(), 1)
    saturation = np.median(image_hsv[..., 1][mask]) / 255
    value = np.median(image_hsv[..., 2][mask]) / 255
    if ellipse_iou < 0.85 or saturation > 0.4 or value < 0.35:
        return 0.0
    return float(ellipse_iou * (1 - saturation) * value)


def auto_select(masks, image_hsv):
    """Greedy best-first, dropping anything that overlaps an already accepted egg."""
    scored = sorted(((egg_score(fill_holes(m), image_hsv), i) for i, m in enumerate(masks)), reverse=True)
    kept = []
    for score, i in scored:
        if score == 0:
            break
        m = fill_holes(masks[i])
        if all((m & k).sum() < 0.1 * min(m.sum(), k.sum()) for k in kept):
            kept.append(m)
    return kept


def from_prompts(predictor, image, egg_prompts):
    """One entry per egg: {"pos": [[x, y], ...]} or {"box": [x0, y0, x1, y1]}.
    Boxes are for eggs where a point selects the wrong scale, such as the whole pot."""
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    kept = []
    for prompt in egg_prompts:
        if "box" in prompt:
            masks, _, _ = predictor.predict(box=np.array(prompt["box"], float), multimask_output=False)
            kept.append(fill_holes(masks[0]))
            continue
        points = np.array(prompt["pos"], float)
        masks, scores, _ = predictor.predict(point_coords=points, point_labels=np.ones(len(points)), multimask_output=True)
        kept.append(fill_holes(masks[int(np.argmax(scores))]))
    return kept


def select(only=()):
    """`only`: limit to these image ids, to redo a few after editing their prompts."""
    from segment_anything import SamPredictor
    os.makedirs(SAM_MASKS, exist_ok=True)
    overlays = os.path.join(OUT, "sam_overlays")
    os.makedirs(overlays, exist_ok=True)
    prompts = {}
    if os.path.exists(PROMPTS):
        with open(PROMPTS) as f:
            prompts = {int(k): v for k, v in json.load(f).items()}
    predictor = SamPredictor(load_sam()) if prompts else None

    summary_path = os.path.join(OUT, "sam_summary.json")
    summary = {}
    if only:
        with open(summary_path) as f:
            summary = {int(k): v for k, v in json.load(f).items()}
    for image_id, path in image_paths().items():
        if only and image_id not in only:
            continue
        image = cv2.imread(path)
        if image_id in prompts:
            eggs, source = from_prompts(predictor, image, prompts[image_id]), "point-prompt"
        else:
            data = np.load(os.path.join(CANDIDATES, f"{image_id}.npz"))
            eggs, source = auto_select(data["masks"], cv2.cvtColor(image, cv2.COLOR_BGR2HSV)), "auto"
        labels = np.zeros(image.shape[:2], np.uint8)
        for k, m in enumerate(sorted(eggs, key=lambda m: -m.sum()), 1):
            labels[m & (labels == 0)] = k
        cv2.imwrite(os.path.join(SAM_MASKS, f"{image_id}.png"), labels)
        cv2.imwrite(os.path.join(overlays, f"{image_id}.jpg"), overlay(image, labels))
        summary[image_id] = {"eggs": len(eggs), "source": source}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print("images", len(summary), "| point-prompted", len(prompts),
          "| egg counts", np.unique([s["eggs"] for s in summary.values()], return_counts=True))


if __name__ == "__main__":
    if sys.argv[1] == "generate":
        generate()
    else:
        select({int(i) for i in sys.argv[2:]})
