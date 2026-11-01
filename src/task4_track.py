"""Task 4: track individual eggs through the video.

Eggs look identical, so identity rests on motion alone. The hard part of this video is that
a hand carries each egg across the frame in turn, hiding it and reversing their left-right order.

  Detect     strong: the classical segmenter, tuned to this video's scale
             weak:   any white blob, e.g. the part of an egg showing between fingers
  Associate  (1) Kalman-predicted tracks to strong detections
             (2) leftover tracks to weak detections, close by only (ByteTrack's idea)
             (3) leftover strong detections to lost tracks, nearest last position first
             (4) lost tracks to large white blobs, an egg showing between fingers
  Spawn      a new ID only if a strong detection persists and no lost track can claim it

    python src/task4_track.py
"""
import json
import os
from collections import Counter

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from common import OUT, ROOT
from segment import FINAL, segment, to_channel

VIDEO = os.path.join(ROOT, "data", "video.mp4")
KEYFRAMES = os.path.join(ROOT, "src", "video_keyframes.json")
VIDEO_CONFIG = {
    **FINAL,
    "min_area": 0.004, "max_area": 0.03,  # one egg is 2,500 to 4,000 px of a 640x360 frame
    "max_aspect": 2.0, "keep_ratio": 0.4,  # two eggs are always in frame, so keep the runner-up
    "refine": None,  # GrabCut costs 3 s per image; an ellipse is enough to track with
}
WEAK_LEVEL = 190     # whiteness of a shell in this video; the shirt's white checks are too small to pass
WEAK_MIN_AREA = 150
REACQUIRE_AREA = 500
GATE_STRONG = 60     # px per frame of association radius
GATE_WEAK = 45
CONFIRM_FRAMES = 8
PALETTE = {1: (60, 60, 255), 2: (60, 220, 60), 3: (255, 160, 30), 4: (0, 220, 240)}


def detect(frame):
    whiteness = to_channel(cv2.medianBlur(frame, 5), "whiteness")
    labels = segment(frame, VIDEO_CONFIG)
    # Absolute test: the proposal levels are relative to each frame, so table reflections
    # and the title cards' vignette would otherwise pass as well-shaped ellipses.
    strong = [m for m in (labels == i for i in range(1, labels.max() + 1))
              if np.median(whiteness[m]) > WEAK_LEVEL]
    n, cc, stats, _ = cv2.connectedComponentsWithStats((whiteness > WEAK_LEVEL).astype(np.uint8), connectivity=8)
    weak = [cc == i for i in range(1, n) if stats[i, 4] >= WEAK_MIN_AREA
            and not any((cc == i)[s].any() for s in strong)]
    return strong, weak


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()], np.float32)


class Track:
    def __init__(self, track_id, mask, frame_index):
        self.id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 10
        c = centroid(mask)
        self.kf.statePost = np.array([[c[0]], [c[1]], [0], [0]], np.float32)
        self.last_seen = c
        self.predicted = c
        self.mask = mask
        self.area = int(mask.sum())
        self.hits = 1
        self.misses = 0
        self.confirmed = False
        self.born = frame_index
        self.trail = [tuple(c)]

    def predict(self):
        self.predicted = self.kf.predict()[:2].ravel()
        # A hidden egg isn't coasting at its last velocity, it has stopped or is in a hand.
        if self.misses > 3:
            self.predicted = self.last_seen
            self.kf.statePost[2:] = 0

    def update(self, mask, partial=False):
        c = centroid(mask)
        if partial:  # the centroid of a sliver isn't the egg's, so trust it only loosely
            c = 0.5 * c + 0.5 * self.predicted
        else:
            self.area = int(mask.sum())
        self.kf.correct(c.reshape(2, 1))
        self.last_seen = c
        self.mask = mask
        self.hits += 1
        self.misses = 0
        self.trail.append(tuple(c))
        self.confirmed = self.confirmed or self.hits >= CONFIRM_FRAMES


def hungarian(tracks, masks, gate):
    """Pairs (track index, mask index) within the gate, closest overall."""
    if not tracks or not masks:
        return []
    cost = np.array([[np.linalg.norm(t.predicted - centroid(m)) for m in masks] for t in tracks])
    return [(r, c) for r, c in zip(*linear_sum_assignment(cost)) if cost[r, c] <= gate]


def track(frames):
    """Yields (frame index, frame, confirmed tracks) for each frame."""
    tracks, next_id = [], 1
    for index, frame in frames:
        strong, weak = detect(frame)
        for t in tracks:
            t.predict()
        updated = set()

        # (1) visible or recently visible tracks to strong detections
        recent = [t for t in tracks if t.misses <= 3]
        used = set()
        for r, c in hungarian(recent, strong, GATE_STRONG):
            recent[r].update(strong[c])
            updated.add(recent[r])
            used.add(c)

        # (2) still-unmatched confirmed tracks to nearby white slivers
        rest = [t for t in tracks if t.confirmed and t not in updated]
        for r, c in hungarian(rest, weak, GATE_WEAK):
            rest[r].update(weak[c], partial=True)
            updated.add(rest[r])

        # (3) leftover strong detections: hidden eggs coming back, else maybe a new egg
        lost = [t for t in tracks if t.confirmed and t not in updated]
        leftovers = [strong[c] for c in range(len(strong)) if c not in used]
        for mask in sorted(leftovers, key=lambda m: -m.sum()):
            if lost:
                nearest = min(lost, key=lambda t: np.linalg.norm(t.last_seen - centroid(mask)))
                nearest.update(mask)
                updated.add(nearest)
                lost.remove(nearest)
            else:
                pending = [t for t in tracks if not t.confirmed and t not in updated]
                match = hungarian(pending, [mask], GATE_STRONG)
                if match:
                    pending[match[0][0]].update(mask)
                    updated.add(pending[match[0][0]])
                else:
                    born = Track(next_id, mask, index)
                    tracks.append(born)
                    updated.add(born)
                    next_id += 1

        # (4) a lost egg showing between fingers won't pass as an ellipse but is a big white blob
        lost = [t for t in lost if t not in updated]
        big = [m for m in weak if m.sum() >= REACQUIRE_AREA and not any((m & t.mask).any() for t in updated)]
        for mask in sorted(big, key=lambda m: -m.sum()):
            if lost:
                nearest = min(lost, key=lambda t: np.linalg.norm(t.last_seen - centroid(mask)))
                nearest.update(mask, partial=False)
                updated.add(nearest)
                lost.remove(nearest)

        for t in tracks:
            if t not in updated:
                t.misses += 1
        tracks = [t for t in tracks if t.confirmed or t.misses < 5]
        yield index, frame, [t for t in tracks if t.confirmed]


def draw(frame, tracks):
    out = frame.copy()
    for t in tracks:
        color = PALETTE.get(t.id, (255, 255, 255))
        if t.misses == 0:
            out[t.mask] = 0.5 * out[t.mask] + 0.5 * np.array(color)
            contours, _ = cv2.findContours(t.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 1)
        x, y = int(t.last_seen[0]), int(t.last_seen[1])
        label = f"egg {t.id}" + ("" if t.misses == 0 else " (hidden)")
        cv2.putText(out, label, (x - 25, y - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x - 25, y - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        trail = np.array(t.trail[-48:], np.int32)
        if len(trail) > 1:
            cv2.polylines(out, [trail], False, color, 1, cv2.LINE_AA)
    return out


def main():
    capture = cv2.VideoCapture(VIDEO)
    fps = capture.get(cv2.CAP_PROP_FPS)
    size = (int(capture.get(3)), int(capture.get(4)))
    os.makedirs(OUT, exist_ok=True)
    writer = cv2.VideoWriter(os.path.join(OUT, "tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def frames():
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield index, frame
            index += 1

    with open(KEYFRAMES) as f:
        keyframes = {int(k): v for k, v in json.load(f).items()}
    log, snapshots = [], {}
    for index, frame, tracks in track(frames()):
        visible = {t.id: t.last_seen.tolist() for t in tracks if t.misses == 0}
        log.append({"frame": index, "tracks": visible,
                    "hidden": [t.id for t in tracks if t.misses > 0]})
        rendered = draw(frame, tracks)
        writer.write(rendered)
        if index in keyframes:
            snapshots[index] = rendered
    writer.release()
    with open(os.path.join(OUT, "tracks.json"), "w") as f:
        json.dump(log, f)
    report(log, keyframes, snapshots)


def report(log, keyframes, snapshots):
    """Score identity against hand-labelled egg positions. A/B map to whichever IDs fit best."""
    by_frame = {e["frame"]: e["tracks"] for e in log}
    ids = sorted({int(i) for e in log for i in e["tracks"]})
    print("track ids:", ids, "| frames with any egg:", sum(1 for e in log if e["tracks"]))
    tally = {}
    for frame, truth in keyframes.items():
        for egg, position in truth.items():
            near = [int(i) for i, p in by_frame.get(frame, {}).items()
                    if np.hypot(p[0] - position[0], p[1] - position[1]) < 50]
            tally.setdefault(egg, []).append((frame, near))
    mapping = {}
    for egg, seen in tally.items():
        counts = Counter(i for _, near in seen for i in near)
        mapping[egg] = counts.most_common(1)[0][0] if counts else None
    correct = wrong = absent = 0
    for egg, seen in tally.items():
        for frame, near in seen:
            if mapping[egg] in near:
                correct += 1
            elif near:
                wrong += 1
                print(f"  WRONG frame {frame}: egg {egg} carries id {near}")
            else:
                absent += 1
                print(f"  no track at frame {frame} for egg {egg}")
    print(f"mapping {mapping}: correct {correct}, wrong id {wrong}, no track {absent}, of {correct + wrong + absent}")
    with open(os.path.join(OUT, "tracking_score.json"), "w") as f:
        json.dump({"mapping": mapping, "correct": correct, "wrong_id": wrong, "no_track": absent,
                   "track_ids": ids}, f, indent=1)
    keys = sorted(snapshots)
    tiles = [cv2.resize(snapshots[k], (320, 180)) for k in keys]
    for k, tile in zip(keys, tiles):
        cv2.putText(tile, str(k), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(OUT, "tracking_sheet.jpg"),
                np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == "__main__":
    main()
