"""Paths and helpers shared by every task."""
import glob
import json
import os
import re

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "outputs")
SAM_MASKS = os.path.join(OUT, "sam_masks")
SPLIT_FILE = os.path.join(ROOT, "splits", "split.json")


def image_paths():
    """{image id: path}, across Roboflow's train/valid/test folders (its split is ignored)."""
    paths = glob.glob(os.path.join(RAW, "*", "*", "*.jpg"))
    return dict(sorted(
        ((int(re.search(r"not_damaged_(\d+)_jpg", p).group(1)), p) for p in paths)))


def load_split():
    with open(SPLIT_FILE) as f:
        return json.load(f)


def load_gt(image_id):
    """Instance label map from Task 1: 0 is background, 1..n are eggs."""
    return cv2.imread(os.path.join(SAM_MASKS, f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)


def overlay(image, labels, alpha=0.55):
    """Paint each instance in its own colour with a thin white outline."""
    palette = [(40, 40, 255), (40, 200, 40), (255, 120, 30), (0, 210, 240), (220, 60, 220)]
    out = image.copy()
    for i in np.unique(labels):
        if i == 0:
            continue
        m = labels == i
        out[m] = (1 - alpha) * out[m] + alpha * np.array(palette[(int(i) - 1) % len(palette)])
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out
