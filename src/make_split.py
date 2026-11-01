"""Task 2: 4:1 dev/eval split with whole scenes on one side, to limit leakage.

The photos are bursts: same surface, same eggs, same lighting, slightly different framing.
A per-image split (Roboflow's included) would put near-copies on both sides. Scenes below
were assigned by eye from a contact sheet, then checked by cross-split perceptual hashing.
"""
import itertools
import json
import os

import cv2
import imagehash
import numpy as np
from PIL import Image

from common import SPLIT_FILE, image_paths, load_gt

SCENES = {
    "hand_dark_floor": [1, 2, 20, 24, 25, 44],
    "striped_towel": [3, 4, 5, 6],
    "leaves": [10, 11, 12, 13, 49],
    "gravel": [14, 27, 36, 41, 50, 78],
    "purple_fabric": [15, 37, 55, 95],
    "carpet_earphones": [16, 52, 62],
    "keyboard": [17, 18, 19, 43],
    "pink_bowl": [21, 22, 34, 35, 39, 40],
    "concrete": [28, 29, 33, 45, 146],
    "white_tray": [30, 32, 42, 116, 117],
    "tiles": [31, 149, 150],
    "white_bag": [104, 106, 109, 111, 112, 123, 125, 143],
    "marble": [100, 101, 102, 103, 105, 115, 121, 132, 133, 134, 136, 140, 142, 144, 148],
    "rice": [107, 122, 126],
    "orange_tub": [108, 110, 120, 127, 130, 139, 141],
    "steel_pot": [113, 114, 124, 128, 129, 135, 145, 147, 153, 159],
    "foil": [118, 119, 131, 138],
    "hand_light": [151, 152, 155, 157, 158, 160, 161],
    "green_carpet": [154, 162, 164],
    "blue_bucket": [156, 163],
}
EVAL_SIZE = 22  # 110 / 5


def choose_eval_scenes():
    """Exhaustive search for a 22-image scene subset that best matches the dataset's
    multi-egg rate and egg size, so that eval is neither easier nor harder by construction."""
    ids = [i for v in SCENES.values() for i in v]
    assert sorted(ids) == sorted(image_paths()), "every image must be in exactly one scene"
    gt = {i: load_gt(i) for i in ids}
    multi = {i: int(gt[i].max() > 1) for i in ids}
    size = {i: float((gt[i] > 0).mean()) for i in ids}
    all_multi, all_size = np.mean(list(multi.values())), np.mean(list(size.values()))

    names, best, best_cost = list(SCENES), None, np.inf
    for r in range(1, len(names)):
        for combo in itertools.combinations(names, r):
            chosen = [i for n in combo for i in SCENES[n]]
            if len(chosen) != EVAL_SIZE:
                continue
            cost = (abs(np.mean([multi[i] for i in chosen]) - all_multi)
                    + abs(np.mean([size[i] for i in chosen]) - all_size))
            if cost < best_cost:
                best, best_cost = combo, cost
    return sorted(best)


def main():
    eval_scenes = choose_eval_scenes()
    dev_scenes = sorted(set(SCENES) - set(eval_scenes))
    split = {
        "dev_scenes": dev_scenes,
        "eval_scenes": eval_scenes,
        "dev": sorted(i for n in dev_scenes for i in SCENES[n]),
        "eval": sorted(i for n in eval_scenes for i in SCENES[n]),
    }
    assert not set(split["dev"]) & set(split["eval"])
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    with open(SPLIT_FILE, "w") as f:
        json.dump(split, f, indent=1)

    paths = image_paths()
    hashes = {i: imagehash.phash(Image.open(p), hash_size=8) for i, p in paths.items()}
    scene_of = {i: n for n, v in SCENES.items() for i in v}
    pairs = sorted((hashes[a] - hashes[b], a, b) for a in split["dev"] for b in split["eval"])
    within = [hashes[a] - hashes[b] for a, b in itertools.combinations(paths, 2)
              if scene_of[a] == scene_of[b]]
    print(f"dev {len(split['dev'])} ({len(dev_scenes)} scenes), eval {len(split['eval'])}: {eval_scenes}")
    print("closest cross-split pHash pairs (distance, dev, eval):", pairs[:5])
    print(f"within-scene pHash distance: median {np.median(within):.0f}, min {min(within)}")
    gt = {i: load_gt(i).max() for i in paths}
    for part in ("dev", "eval"):
        print(part, "egg counts", dict(zip(*np.unique([gt[i] for i in split[part]], return_counts=True))))


if __name__ == "__main__":
    main()
