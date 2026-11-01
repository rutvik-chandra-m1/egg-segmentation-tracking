"""Classical egg segmentation: no learned weights, every step configurable.

Pipeline: denoise -> channel -> region proposals -> egg-likeness selection -> refinement.
task2_experiments.py walks through configurations; FINAL at the bottom is the one kept.
"""
import cv2
import numpy as np
from skimage.segmentation import felzenszwalb

BASE = {
    "denoise": None,              # None | "median" | "bilateral"
    "channel": "gray",            # gray | l | v | whiteness | smooth | pale, or a list of them
    "proposals": ["otsu"],        # any of otsu, adaptive, sweep, sweep_dark, felz
    "sweep_levels": 12,
    "felz_scales": (200, 500),
    "open": 0,                    # morphological kernel sizes, 0 is off
    "close": 0,
    "selection": "all",           # all | largest | egg | edge
    "min_area": 0.004,            # fractions of the image
    "max_area": 0.6,
    "min_ellipse_iou": 0.85,
    "max_aspect": 2.2,
    "max_saturation": 1.0,
    "contrast_weight": 0.0,
    "keep_ratio": 0.75,           # keep eggs scoring within this ratio of the best one
    "refine": None,               # None | "grabcut"
    "watershed": False,
    "fill_holes": True,
    "edge_blur": 1.5,             # edge selection only, from here down
    "canny": (20, 60),
    "edge_thresh": 2.5,           # gradient, in gray levels per pixel, that counts as boundary
    "arc_pairs": True,
    "region_seeds": False,        # also fit ellipses to the threshold proposals
    "interior_weight": 0.0,       # penalty for edges inside the ellipse
    "min_score": 0.35,
    "size_weight": 0.0,
    "saturation_weight": 0.0,     # penalty for colour inside the ellipse
    "max_arcs": 40,
    "same_size_ratio": None,      # lower keep ratio for ellipses the size of the best one
    "variance_weight": 0.0,       # penalty for uneven brightness inside the ellipse
    "polish": False,              # re-fit each ellipse to the edge pixels near it
    "grabcut_band": 0.08,         # undecided band as a fraction of the egg's size
}


def to_channel(image, name):
    if name == "gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if name == "v":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    if name == "pale":  # inverse saturation
        return 255 - cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    if name == "smooth":  # inverse local contrast: shells are smooth, cloth, ash and leaves are not
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean = cv2.blur(gray, (9, 9))
        deviation = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mean * mean, 0))
        return 255 - np.clip(cv2.blur(deviation, (9, 9)) * 12, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if name == "l":
        return lab[..., 0].astype(np.uint8)
    if name == "whiteness":  # bright and colourless, which shells are and skin or fabric are not
        chroma = np.abs(lab[..., 1] - 128) + np.abs(lab[..., 2] - 128)
        return np.clip(lab[..., 0] - 2 * chroma, 0, 255).astype(np.uint8)
    raise ValueError(name)


def binary_proposals(channel, cfg):
    """Each proposal method yields whole-image binary masks."""
    methods = cfg["proposals"]
    if "otsu" in methods:
        yield cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    if "adaptive" in methods:
        yield cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 151, -5) > 0
    if "sweep" in methods or "sweep_dark" in methods:
        levels = np.linspace(*np.percentile(channel, (5, 99)), cfg["sweep_levels"])
        for level in levels:
            if "sweep" in methods:
                yield channel > level
            if "sweep_dark" in methods:
                yield channel < level


def morph(mask, cfg):
    out = mask.astype(np.uint8)
    if cfg["open"]:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["open"],) * 2))
    if cfg["close"]:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["close"],) * 2))
    return out


def filled_components(binary, cfg):
    """Connected components within the area limits, holes filled, as (x, y, mask crop) tuples."""
    total = binary.size
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if cfg["min_area"] * total <= area <= cfg["max_area"] * total:
            contours, _ = cv2.findContours((labels[y:y + h, x:x + w] == i).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            crop = np.zeros((h, w), np.uint8)
            if cfg["fill_holes"]:
                cv2.drawContours(crop, contours, -1, 1, -1)
            else:
                crop[labels[y:y + h, x:x + w] == i] = 1
            yield x, y, crop


def ellipse_fit(x, y, crop):
    """(ellipse IoU, aspect ratio, ellipse) of a component, or None if it has no usable contour."""
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    if len(contour) < 20:
        return None
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (d1, d2), angle = ellipse
    if min(d1, d2) < 1 or max(d1, d2) > 2 * sum(crop.shape):  # near-collinear contours fit absurd ellipses
        return None
    pad = int(max(d1, d2))
    canvas = np.zeros((crop.shape[0] + 2 * pad, crop.shape[1] + 2 * pad), np.uint8)
    cv2.ellipse(canvas, ((cx + pad, cy + pad), (d1, d2), angle), 1, -1)
    fitted = canvas > 0
    region = np.zeros_like(fitted)
    region[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop > 0
    iou = (fitted & region).sum() / max((fitted | region).sum(), 1)
    return iou, max(d1, d2) / min(d1, d2), ((cx + x, cy + y), (d1, d2), angle)


def egg_score(x, y, crop, gray, hsv, cfg):
    """Score in (0, 1], or 0 if the region can't be an egg."""
    fit = ellipse_fit(x, y, crop)
    if fit is None or fit[0] < cfg["min_ellipse_iou"] or fit[1] > cfg["max_aspect"]:
        return 0.0, None
    h, w = crop.shape
    mask = crop > 0
    saturation = np.median(hsv[y:y + h, x:x + w, 1][mask]) / 255
    if saturation > cfg["max_saturation"]:
        return 0.0, None
    score = float(fit[0] ** 4) * (1 - saturation) * (0.5 + 0.5 * mask.sum() / gray.size ** 0.5 / 200)
    if cfg["contrast_weight"]:
        # Egg interior against a ring just outside: a true object boundary separates them.
        r = max(5, int(0.12 * max(fit[2][1])))
        x0, y0 = max(x - r, 0), max(y - r, 0)
        x1, y1 = min(x + w + r, gray.shape[1]), min(y + h + r, gray.shape[0])
        full = np.zeros((y1 - y0, x1 - x0), np.uint8)
        full[y - y0:y - y0 + h, x - x0:x - x0 + w] = crop
        ring = cv2.dilate(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
        ring &= full == 0
        if ring.any():
            inner = gray[y:y + h, x:x + w][mask].mean()
            score *= 1 + cfg["contrast_weight"] * abs(inner - gray[y0:y1, x0:x1][ring].mean()) / 255
    return score, fit[2]


def edge_arcs(edges, max_arcs=40, min_length=30):
    """Split edge contours at sharp corners into smooth arcs, longest first."""
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    arcs, step = [], 6
    for contour in contours:
        points = contour[:, 0].astype(np.float32)
        if len(points) < min_length:
            continue
        chord = points[step:] - points[:-step]
        angle = np.arctan2(chord[:, 1], chord[:, 0])
        turn = np.abs(np.angle(np.exp(1j * (angle[step:] - angle[:-step]))))
        cuts = np.nonzero(turn > 1.0)[0] + step
        for piece in np.split(points, cuts):
            if len(piece) >= min_length:
                arcs.append(piece)
    return sorted(arcs, key=len, reverse=True)[:max_arcs]


def ellipse_support(ellipse, gx, gy, thresh):
    """Fraction of the in-frame perimeter sitting on a gradient normal to the ellipse."""
    (cx, cy), (d1, d2), angle = ellipse
    a, b, theta = d1 / 2, d2 / 2, np.deg2rad(angle)
    t = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    c, s = np.cos(theta), np.sin(theta)
    px, py = a * np.cos(t), b * np.sin(t)
    x, y = cx + px * c - py * s, cy + px * s + py * c
    nx, ny = np.cos(t) / a, np.sin(t) / b
    nx, ny = nx * c - ny * s, nx * s + ny * c
    norm = np.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    h, w = gx.shape
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    if inside.mean() < 0.6:  # eggs cut by the frame are fine, mostly-outside fits are not
        return 0.0
    x, y, nx, ny = x[inside], y[inside], nx[inside], ny[inside]
    best = np.zeros(len(x))
    for offset in (-3, -1.5, 0, 1.5, 3):  # real eggs are not perfect ellipses
        xi = np.clip((x + offset * nx).astype(int), 0, w - 1)
        yi = np.clip((y + offset * ny).astype(int), 0, h - 1)
        best = np.maximum(best, np.abs(gx[yi, xi] * nx + gy[yi, xi] * ny))
    return float((best > thresh).mean())


def edge_ellipses(gray, cfg, seeds=()):
    """Ellipses fitted to single arcs and pairs of arcs, plus any seed ellipses."""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), cfg["edge_blur"]), *cfg["canny"])
    arcs = edge_arcs(edges, cfg["max_arcs"])
    groups = list(arcs)
    if cfg["arc_pairs"]:
        groups += [np.vstack((p, q)) for i, p in enumerate(arcs) for q in arcs[i + 1:]]
    ellipses = list(seeds)
    for points in groups:
        if len(points) >= 5:
            ellipses.append(cv2.fitEllipse(points))
    return ellipses, edges


def polish(ellipse, edges, gx, gy, thresh, band=7, rounds=2):
    """Re-fit to edge pixels hugging the ellipse: arcs give a rough fit, this tightens it."""
    best, best_support = ellipse, ellipse_support(ellipse, gx, gy, thresh)
    for _ in range(rounds):
        ring = np.zeros(edges.shape, np.uint8)
        cv2.ellipse(ring, best, 1, 2 * band)
        ys, xs = np.nonzero(ring & (edges > 0))
        if len(xs) < 20:
            break
        fitted = cv2.fitEllipse(np.column_stack((xs, ys)).astype(np.float32))
        if not np.isfinite(fitted[1]).all():
            break
        support = ellipse_support(fitted, gx, gy, thresh)
        if support < best_support:
            break
        best, best_support = fitted, support
    return best


def ellipse_mask(ellipse, shape):
    canvas = np.zeros(shape, np.uint8)
    cv2.ellipse(canvas, ellipse, 1, -1)
    return canvas > 0


def select_by_edges(gray, hsv, cfg, seeds):
    """Rank candidate ellipses by edge support, then drop overlapping ones."""
    blurred = cv2.GaussianBlur(gray, (0, 0), cfg["edge_blur"]).astype(np.float32)
    gx, gy = cv2.Sobel(blurred, cv2.CV_32F, 1, 0) / 8, cv2.Sobel(blurred, cv2.CV_32F, 0, 1) / 8
    candidates, edges = edge_ellipses(gray, cfg, seeds)
    total = gray.size
    ranked = []
    for ellipse in candidates:
        _, (d1, d2), _ = ellipse
        if not np.isfinite([d1, d2]).all() or min(d1, d2) < 10:
            continue
        area = np.pi * d1 * d2 / 4 / total
        if not cfg["min_area"] <= area <= cfg["max_area"] or max(d1, d2) / min(d1, d2) > cfg["max_aspect"]:
            continue
        support = ellipse_support(ellipse, gx, gy, cfg["edge_thresh"])
        ranked.append((support * (1 + cfg["size_weight"] * np.sqrt(area)), ellipse))
    ranked.sort(key=lambda r: -r[0])

    rescored = []
    for support, ellipse in ranked[:60]:
        if cfg["polish"]:
            ellipse = polish(ellipse, edges, gx, gy, cfg["edge_thresh"])
        mask = ellipse_mask(ellipse, gray.shape)
        if not mask.any():  # would score NaN, which passes every threshold below
            continue
        if cfg["interior_weight"]:
            inner = cv2.erode(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            density = (edges[inner] > 0).mean() if inner.any() else 1.0
            support *= max(1 - cfg["interior_weight"] * density, 0)
        if cfg["saturation_weight"]:
            support *= max(1 - cfg["saturation_weight"] * np.median(hsv[..., 1][mask]) / 255, 0)
        if cfg["variance_weight"]:
            support *= max(1 - cfg["variance_weight"] * gray[mask].std() / 64, 0)
        rescored.append((support, mask))
    rescored.sort(key=lambda r: -r[0])

    kept = []
    for support, mask in rescored:
        if support < cfg["min_score"]:
            break
        if kept:
            ratio = cfg["keep_ratio"]
            # Eggs in one photo are alike in size, junk usually is not, so a same-size
            # ellipse has to clear a lower bar than an arbitrary one.
            if cfg["same_size_ratio"] and 0.5 <= mask.sum() / kept[0].sum() <= 2:
                ratio = cfg["same_size_ratio"]
            if support < ratio * rescored[0][0]:
                continue
        if all((mask & k).sum() < 0.1 * min(mask.sum(), k.sum()) for k in kept):
            kept.append(mask)
    return kept


def grabcut(image, mask, band=0.08):
    """Snap a rough region to the image edges, with a band around its boundary left undecided."""
    size = max(3, int(band * np.sqrt(mask.sum())) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    trimap = np.full(mask.shape, cv2.GC_BGD, np.uint8)
    trimap[cv2.dilate(mask.astype(np.uint8), kernel) > 0] = cv2.GC_PR_BGD
    trimap[mask] = cv2.GC_PR_FGD
    trimap[cv2.erode(mask.astype(np.uint8), kernel) > 0] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(image, trimap, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:  # GMM can't fit a nearly uniform region
        return mask
    out = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
    contours, _ = cv2.findContours(out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return filled > 0


def split_touching(mask):
    """Distance-transform watershed, for blobs that hold more than one egg."""
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    peaks = (distance > 0.6 * distance.max()).astype(np.uint8)
    n, markers = cv2.connectedComponents(peaks)
    if n <= 2:
        return [mask]
    markers = markers + 1
    markers[(mask > 0) & (peaks == 0)] = 0
    canvas = cv2.cvtColor(np.clip(distance * 255 / distance.max(), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    result = cv2.watershed(255 - canvas, markers.astype(np.int32))
    return [result == k for k in range(2, n + 1)]


def labels_from(masks, shape):
    labels = np.zeros(shape, np.uint8)
    for k, m in enumerate(sorted(masks, key=lambda m: -m.sum()), 1):
        labels[m & (labels == 0)] = k
    return labels


def segment(image, config=None):
    """Returns an instance label map: 0 is background, 1..n are eggs."""
    cfg = {**BASE, **(config or {})}
    smooth = image
    if cfg["denoise"] == "median":
        smooth = cv2.medianBlur(image, 7)
    elif cfg["denoise"] == "bilateral":
        smooth = cv2.bilateralFilter(image, 9, 50, 7)
    names = cfg["channel"] if isinstance(cfg["channel"], list) else [cfg["channel"]]
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(smooth, cv2.COLOR_BGR2HSV)
    shape = image.shape[:2]

    binaries = [morph(b, cfg) for name in names for b in binary_proposals(to_channel(smooth, name), cfg)]
    if "felz" in cfg["proposals"]:
        for scale in cfg["felz_scales"]:
            segments = felzenszwalb(smooth[..., ::-1], scale=scale, sigma=0.8, min_size=100)
            binaries += [(segments == s).astype(np.uint8) for s in np.unique(segments)]

    candidates = [c for b in binaries for c in filled_components(b, cfg)]

    if cfg["selection"] == "edge":
        seeds = []
        if cfg["region_seeds"]:
            fits = (ellipse_fit(*c) for c in candidates)
            seeds = [f[2] for f in fits if f is not None]
        kept = select_by_edges(gray, hsv, cfg, seeds)
        if cfg["refine"] == "grabcut":
            kept = [grabcut(smooth, m, cfg["grabcut_band"]) for m in kept]
        return labels_from(kept, shape)

    if cfg["selection"] == "all":
        union = np.zeros(shape, np.uint8)
        for b in binaries:
            union |= b
        return labels_from([union > 0], shape)

    def paste(x, y, crop):
        full = np.zeros(shape, bool)
        full[y:y + crop.shape[0], x:x + crop.shape[1]] = crop > 0
        return full

    if not candidates:
        return np.zeros(shape, np.uint8)
    if cfg["selection"] == "largest":
        kept = [paste(*max(candidates, key=lambda c: c[2].sum()))]
    else:
        scored = sorted(((*egg_score(*c, gray, hsv, cfg), i) for i, c in enumerate(candidates)),
                        key=lambda t: -t[0])
        kept = []
        for score, ellipse, i in scored:
            if score == 0 or (kept and score < cfg["keep_ratio"] * scored[0][0]):
                break
            m = paste(*candidates[i])
            if all((m & k).sum() < 0.1 * min(m.sum(), k.sum()) for k in kept):
                kept.append(m)

    if cfg["watershed"]:
        kept = [part for m in kept for part in split_touching(m)]
    if cfg["refine"] == "grabcut":
        kept = [grabcut(smooth, m) for m in kept]
    return labels_from(kept, shape)


# Trial T32 of the experiment log, the best on the dev split, frozen before eval was run.
FINAL = {
    "selection": "edge", "denoise": "median", "proposals": [], "min_area": 0.01, "max_area": 0.45,
    "canny": (10, 30), "edge_thresh": 1.5, "max_arcs": 80,
    "interior_weight": 4.0, "saturation_weight": 1.5, "keep_ratio": 0.9,
    "refine": "grabcut", "grabcut_band": 0.1,
}
