"""Segmentation metrics, comparing two instance label maps (0 = background)."""
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment


def _boundary(mask):
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def boundary_metrics(pred, gt, tol=3):
    """Boundary F1 at `tol` px, plus 95th percentile symmetric boundary distance (HD95)."""
    bp, bg = _boundary(pred), _boundary(gt)
    if not bp.any() or not bg.any():
        return 0.0, float(np.hypot(*gt.shape))
    d_to_g = distance_transform_edt(~bg)[bp]
    d_to_p = distance_transform_edt(~bp)[bg]
    precision, recall = (d_to_g <= tol).mean(), (d_to_p <= tol).mean()
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(f1), float(np.percentile(np.r_[d_to_g, d_to_p], 95))


def instance_metrics(pred_labels, gt_labels, iou_thresh=0.5):
    """Hungarian matching on IoU. Returns TP, FP, FN and the IoUs of the matched pairs.
    Strictly above 0.5, as in panoptic quality: two equal eggs merged into one mask
    score exactly 0.5 against each and must not count as a match."""
    p_ids = [i for i in np.unique(pred_labels) if i]
    g_ids = [i for i in np.unique(gt_labels) if i]
    if not p_ids or not g_ids:
        return 0, len(p_ids), len(g_ids), []
    ious = np.zeros((len(p_ids), len(g_ids)))
    for a, p in enumerate(p_ids):
        pm = pred_labels == p
        for b, g in enumerate(g_ids):
            gm = gt_labels == g
            ious[a, b] = (pm & gm).sum() / max((pm | gm).sum(), 1)
    rows, cols = linear_sum_assignment(-ious)
    matched = [float(ious[r, c]) for r, c in zip(rows, cols) if ious[r, c] > iou_thresh]
    return len(matched), len(p_ids) - len(matched), len(g_ids) - len(matched), matched


def score(pred_labels, gt_labels):
    pred, gt = pred_labels > 0, gt_labels > 0
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    bf, hd95 = boundary_metrics(pred, gt)
    i_tp, i_fp, i_fn, matched = instance_metrics(pred_labels, gt_labels)
    denominator = i_tp + 0.5 * i_fp + 0.5 * i_fn
    return {
        "iou": float(tp / max(tp + fp + fn, 1)),
        "dice": float(2 * tp / max(2 * tp + fp + fn, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "boundary_f1": bf,
        "hd95": hd95,
        "inst_tp": i_tp, "inst_fp": i_fp, "inst_fn": i_fn,
        "count_ok": int(len(np.unique(pred_labels[pred_labels > 0])) == len(np.unique(gt_labels[gt_labels > 0]))),
        "pq": float(sum(matched) / denominator) if denominator else 0.0,
    }


def _self_check():
    gt = np.zeros((100, 100), np.uint8)
    gt[10:30, 10:30], gt[60:80, 60:80] = 1, 2
    same = score(gt, gt)
    assert same["iou"] == 1 and same["boundary_f1"] == 1 and same["pq"] == 1 and same["hd95"] == 0
    # Same pixels as gt but merged into one instance: semantic metrics are perfect,
    # instance metrics must not be. This is the failure mode for touching eggs.
    merged = score((gt > 0).astype(np.uint8), gt)
    assert merged["iou"] == 1 and merged["count_ok"] == 0 and merged["inst_tp"] == 0
    half = np.zeros_like(gt)
    half[10:30, 10:20] = 1
    half_score = score(half, (gt == 1).astype(np.uint8))
    assert abs(half_score["iou"] - 0.5) < 1e-9 and half_score["recall"] == 0.5 and half_score["precision"] == 1
    assert score(np.zeros_like(gt), gt)["iou"] == 0


if __name__ == "__main__":
    _self_check()
    print("metrics self-check ok")
