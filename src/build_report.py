"""Build report/report.pdf from the result files, so its numbers always match the last run.

    python src/build_report.py --github <repo url> --video <google drive url>
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from common import OUT, ROOT, load_split

EVAL = os.path.join(OUT, "eval")
INK, MUTED, RULE, ACCENT = "#1f2933", "#6b7785", "#d5dae0", "#2563a8"

# Written after reading each trial's result: status, then why.
DECISIONS = {
    "T01": ("Baseline", "Recall 0.87 at precision 0.32: bright pixels are mostly background."),
    "T02": ("Discarded", "IoU fell. The largest bright region is often the background, not the egg."),
    "T03": ("Kept", "Almost no IoU change, but median blur stayed useful. Morphology was dropped with region methods."),
    "T04": ("Discarded", "No gain over gray."),
    "T05": ("Kept", "Best so far: colour rejects hands and fabric."),
    "T06": ("Discarded", "Slightly higher IoU, but recall fell to 0.43."),
    "T07": ("Discarded", "Worst trial. One threshold rarely gives a clean ellipse. The idea was sound and came back in T08."),
    "T08": ("Kept", "Largest gain so far: many levels give the ellipse test real choices."),
    "T09": ("Discarded", "Unchanged IoU at 1.6 times the runtime."),
    "T10": ("Kept", "IoU fell, but recall and HD95 improved greatly, as fewer images returned nothing."),
    "T11": ("Kept", "Solves white-on-white regions. Boundary F1 of 0.018 shows the filter window displaces the edge."),
    "T12": ("Kept", "Best region result: channels fail on different images."),
    "T13": ("Discarded", "More proposals scored worse, showing the selector, not proposals, was now the limit."),
    "T14": ("Discarded", "Below T12 at about 7 times its runtime."),
    "T15": ("Superseded", "Fixed the hole-filling flaw found by visual review. Best region method, replaced by T16."),
    "T16": ("Kept", "Change of approach. Boundary F1 more than doubled against T15."),
    "T17": ("Kept", "Large gain: a clean interior separates shells from textured backgrounds."),
    "T18": ("Discarded", "Better counts, but lower IoU: review showed it choosing tubs and bowls."),
    "T19": ("Discarded", "Better boundaries, but it amplified wrong selections. Revisited in T30."),
    "T20": ("Kept", "Small IoU gain, larger count gain."),
    "T21": ("Inconclusive", "No gain alone. T26 later showed it does help with more arcs."),
    "T22": ("Kept", "Large gain: faint outlines now become arcs at all."),
    "T23": ("Kept", "Slightly below T22 alone, kept to test with more arcs."),
    "T24": ("Kept", "Best IoU so far, though count accuracy fell."),
    "T25": ("Discarded", "Worse: poor seeds out-scored good arcs."),
    "T26": ("Discarded", "Below T24, which disproved my T21 reading: the low threshold is worth keeping."),
    "T27": ("Kept", "Count accuracy up with little IoU cost."),
    "T28": ("Discarded", "Lost real eggs: IoU and recall both fell."),
    "T29": ("Discarded", "No gain for 47% more runtime."),
    "T30": ("Superseded", "GrabCut now helps, as selection is right more often. Beaten by T31."),
    "T31": ("Kept", "Best boundary F1 to this point: the wider band beat T30."),
    "T32": ("FINAL", "Best IoU, boundary F1 and count of all trials. PQ tied with T40."),
    "T33": ("Discarded", "Lower on every headline metric."),
    "T34": ("Discarded", "Over-penalises eggs with ash or speckle."),
    "T35": ("Discarded", "Worse overall, and backlit eggs 20 and 24 join the worst images."),
    "T36": ("Discarded", "Best recall (0.90), but count accuracy fell to 0.80."),
    "T37": ("Discarded", "Flawed test: penalty pushed real eggs under the fixed floor. Retested in T39."),
    "T38": ("Discarded", "Same flaw as T37, so uninformative."),
    "T39": ("Discarded", "Fair retest of T37: still below T32."),
    "T40": ("Discarded", "No gain over T32."),
}
PHASES = [
    ("Thresholding (T01 to T07)", "Global and local thresholds on several channels. Best IoU 0.34. Brightness cannot separate white eggs from white surfaces."),
    ("Region proposals (T08 to T15)", "Threshold sweeps and Felzenszwalb regions, chosen by ellipse fit. Best IoU 0.50. T13 showed more proposals scored worse; visual review traced this to hole-filling and to eggs split along shading lines."),
    ("Edge-based ellipses (T16 to T26)", "Since outlines stay visible even where regions fail, ellipses are fitted to Canny arcs and ranked by perimeter gradient support, with interior penalties. Best IoU 0.76."),
    ("Refinement (T27 to T40)", "A stricter keep rule and GrabCut, which only helped once selection was reliable. T32 reached IoU 0.80. The last eight trials found nothing better, so tuning stopped."),
]


def progress_chart(rows, path):
    ids = [r["id"] for r in rows]
    ious = [float(r["iou"]) for r in rows]
    best, running = 0, []
    for v in ious:
        best = max(best, v)
        running.append(best)
    xs = range(1, len(ids) + 1)
    kept = [i for i, t in enumerate(ids) if DECISIONS[t][0] in ("Kept", "FINAL", "Baseline", "Superseded", "Inconclusive")]
    dropped = [i for i in range(len(ids)) if i not in kept]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.step(xs, running, where="post", color=ACCENT, lw=2, label="Best so far")
    ax.scatter([i + 1 for i in kept], [ious[i] for i in kept], s=26, color=ACCENT, edgecolor="white", lw=0.8, zorder=3, label="Kept")
    ax.scatter([i + 1 for i in dropped], [ious[i] for i in dropped], s=26, marker="x", color=MUTED, lw=1.1, zorder=3, label="Discarded")
    for x in (7.5, 15.5, 26.5):
        ax.axvline(x, color=RULE, lw=0.8)
    for x, text in ((4, "Threshold"), (11.5, "Regions"), (21, "Edge ellipses"), (33.5, "Refinement")):
        ax.text(x, 0.97, text, ha="center", va="top", color=MUTED, fontsize=8)
    ax.annotate("T32 (final)", (32, ious[31]), xytext=(33, 0.68), fontsize=8, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
    ax.set(xlabel="Trial", ylabel="Dev mean IoU", ylim=(0, 1), xlim=(0.5, len(ids) + 0.5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.grid(axis="y", color="#eef0f3", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github", default="(repository link to be added)")
    parser.add_argument("--video", default="(video link to be added)")
    args = parser.parse_args()

    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5,
                          textColor=INK, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=14, leading=18,
                        textColor=INK, spaceBefore=12, spaceAfter=6, keepWithNext=1)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, leading=14,
                        textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1)
    title = ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20, leading=25,
                           textColor=INK, alignment=0, spaceAfter=4)
    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=7, leading=8.6, textColor=INK)
    cell_bold = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    note = ParagraphStyle("note", parent=body, fontSize=8, leading=11, textColor=MUTED)
    width = A4[0] - 40 * mm

    def p(text, style=body):
        return Paragraph(text, style)

    def bullets(items):
        return [Paragraph(i, body, bulletText="•") for i in items]

    def figure(path, w=width):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:
            return Image(path, width=w, height=w * im.height / im.width)

    def table(rows, widths, header=True, align_from=1):
        data = [[p(str(c), cell_bold if header and r == 0 else cell) for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(RULE)),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    with open(os.path.join(ROOT, "src", "trials.json")) as f:
        trials = {t["id"]: t for t in json.load(f)}
    with open(os.path.join(OUT, "experiments.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    with open(os.path.join(OUT, "sam_summary.json")) as f:
        sam = json.load(f)
    with open(os.path.join(ROOT, "src", "task1_prompts.json")) as f:
        prompts = json.load(f)
    with open(os.path.join(EVAL, "instance.json")) as f:
        instance = json.load(f)
    with open(os.path.join(OUT, "tracking_score.json")) as f:
        tracking = json.load(f)
    split = load_split()
    summary = pd.read_csv(os.path.join(EVAL, "summary.csv"), index_col=0)
    scenes = pd.read_csv(os.path.join(EVAL, "by_scene.csv"))
    worst = pd.read_csv(os.path.join(EVAL, "eval_sorted.csv"))
    failures = pd.read_csv(os.path.join(EVAL, "failures.csv"))

    boxes = sum("box" in v[0] for v in prompts.values())
    points = len(prompts) - boxes
    auto = len(sam) - len(prompts)
    t32 = next(r for r in rows if r["id"] == "T32")
    m = summary.loc
    chart = os.path.join(EVAL, "progress.png")
    progress_chart(rows, chart)

    story = [p("CV Intern Task: Egg Segmentation and Tracking", title),
             p(f'Code: <link href="{args.github}" color="{ACCENT}">{args.github}</link><br/>'
               f'Tracking video: <link href="{args.video}" color="{ACCENT}">{args.video}</link>'),
             p("Summary", h1)]
    total = tracking["correct"] + tracking["wrong_id"] + tracking["no_track"]
    story += bullets([
        f"<b>Task 1.</b> SAM (ViT-B) masks for all {len(sam)} images. SAM has no notion of an egg, so an egg-likeness filter picked masks automatically on {auto} images, and I reviewed every result and re-prompted the other {len(prompts)} by hand.",
        f"<b>Task 2.</b> A {len(split['dev'])}/{len(split['eval'])} split by physical scene to limit leakage. {len(rows)} logged trials led to a classical algorithm that fits ellipses to edge arcs and refines them with GrabCut.",
        f"<b>Task 3.</b> Eval mean IoU {m['iou', 'eval']:.3f} (95% CI {m['iou', 'eval_ci_low']:.2f} to {m['iou', 'eval_ci_high']:.2f}), median {instance['median_iou']:.3f}, against {m['iou', 'dev']:.3f} on dev. Eval beating dev reflects an easier scene mix, not better generalisation (section 3.1).",
        f"<b>Task 4.</b> Both eggs tracked through hand carries and occlusion with {len(tracking['track_ids'])} IDs and {tracking['wrong_id']} identity errors on {total} hand-labelled checks.",
    ])

    story += [p("1. Segmentation with a pretrained model", h1),
              p("The dataset has 110 photos, stretched to 640 by 640, showing one to three eggs. I ran SAM's automatic mask generator with ViT-B weights on CPU, using a 10 by 10 prompt grid. That spacing of 64 px is fine for eggs that are all over 100 px wide, and halves the runtime of a 16 by 16 grid."),
              p("SAM is class-agnostic, so masks had to be chosen. A filter scored each candidate on ellipse fit, area, brightness and saturation. Its egg masks were accurate, but it also accepted background objects such as lids and keyboard keys, and found nothing on six images. Since these masks become ground truth, I reviewed every overlay:")]
    story += [table([["Source of final mask", "Images"], ["Automatic selection, accepted on review", auto],
                     ["Hand-placed point prompts", points],
                     ["Hand-drawn box prompts, where a point selected the whole pot", boxes]],
                    [width * 0.8, width * 0.2]), Spacer(1, 6)]
    story += [p("<b>Known limits of this ground truth.</b> SAM's own masks are weak on the three eggs under boiling water (124, 135, 145), whose edges are obscured by bubbles, and image 145 keeps a straight cut along the box prompt. Image 117, an egg behind a rod, is also rough. Eggs under leaves (10 to 13) follow the leaves. Scores on these images partly measure SAM's errors.")]

    story += [p("2. Classical segmentation", h1), p("2.1 Dev and eval split", h2),
              p(f"The photos are bursts of the same surface, eggs and light. Roboflow's own split and any per-image split would put near-copies on both sides, so I ignored it. I grouped images by eye into 20 physical scenes, such as the keyboard or the steel pot, and kept each scene whole. An exhaustive search chose the 22-image subset of scenes closest to the full set in multi-egg rate and egg size.")]
    story += bullets([
        f"<b>Result:</b> {len(split['dev'])} dev images from {len(split['dev_scenes'])} scenes, {len(split['eval'])} eval images from {len(split['eval_scenes'])} scenes ({', '.join(s.replace('_', ' ') for s in split['eval_scenes'])}).",
        "<b>Check:</b> the closest dev and eval images are 18 apart in perceptual hash distance. Pairs within a scene go as low as 4, so no near-duplicates crossed.",
        "<b>Process:</b> every parameter was chosen on dev. Eval was run once, after the algorithm was frozen.",
        "<b>Limits:</b> the same physical eggs may appear in both splits. Eval has no 3-egg image, as nearly all of them are in the marble scene, which cannot be divided.",
    ])
    story += [p("2.2 Final algorithm (T32)", h2)]
    story += bullets([
        "<b>Denoise and edges.</b> Median blur, then Canny with low thresholds (10, 30), so faint white-on-white outlines survive.",
        "<b>Arcs.</b> Split edge contours at sharp corners and keep the 80 longest smooth arcs.",
        "<b>Candidates.</b> Fit an ellipse to every arc and every pair of arcs, as an egg outline is often broken into pieces.",
        "<b>Score.</b> The fraction of the perimeter on a gradient normal to the ellipse, reduced for edges inside it and for colour inside it.",
        "<b>Select.</b> Take the best, then others scoring within 90% of it that do not overlap.",
        "<b>Refine.</b> GrabCut in a band around each ellipse, to follow the true outline and exclude fingers.",
    ])
    story += [p("2.3 How the experiments led here", h2), figure(chart)]
    for name, text in PHASES:
        story.append(p(f"<b>{name}.</b> {text}"))
    story.append(p(f"Values are dev means against the SAM masks; each trial inherits from the one named in its parent column, and every trial can be rerun from <font name='Courier'>src/trials.json</font>. Repeat runs of T32 vary by about 0.001 because GrabCut is randomly initialised. Choosing the best of {len(rows)} trials makes dev scores slightly optimistic.", note))
    story += [PageBreak(), p("2.4 Experiment log", h2)]
    log = [["ID", "Change and why", "Parameters (from parent)", "IoU", "BF1", "Count", "PQ", "Decision"]]
    for r in rows:
        t = trials[r["id"]]
        status, reason = DECISIONS[r["id"]]
        params = ", ".join(f"{k}={v}" for k, v in t["changes"].items()).replace("[", "").replace("]", "").replace("'", "")
        parent = f"from {t['parent']}: " if t.get("parent") else ""
        log.append([r["id"], f"<b>{r['name']}.</b> {t['why']}", parent + params, f"{float(r['iou']):.3f}", f"{float(r['boundary_f1']):.3f}",
                    f"{float(r['count_ok']):.3f}", f"{float(r['pq']):.3f}", f"<b>{status}.</b> {reason}"])
    story += [table(log, [22, 146, 86, 25, 25, 30, 25, 136]), Spacer(1, 3),
              p("IoU: mask overlap. BF1: boundary F1 at 3 px. Count: share of images with the correct number of eggs. PQ: panoptic quality, which also penalises merged or split eggs.", note)]

    story += [PageBreak(), p("3. Evaluation and analysis", h1), p("3.1 Quantitative results", h2)]
    names = {"iou": "IoU", "dice": "Dice", "precision": "Precision", "recall": "Recall", "boundary_f1": "Boundary F1 (3 px)",
             "hd95": "HD95 (px, lower is better)", "pq": "Panoptic quality", "count_ok": "Correct egg count"}
    story.append(table([["Metric", "Dev mean", "Eval mean", "Eval 95% CI"]] + [
        [name, f"{m[k, 'dev']:.3f}", f"{m[k, 'eval']:.3f}", f"{m[k, 'eval_ci_low']:.3f} to {m[k, 'eval_ci_high']:.3f}"]
        for k, name in names.items()], [width * 0.4, width * 0.2, width * 0.2, width * 0.2]))
    story += [Spacer(1, 6), p(f"At instance level, with a match needing IoU above 0.5, eval has {instance['tp']} true positives, {instance['fp']} false positives and {instance['fn']} false negatives (precision and recall {instance['precision']:.2f}). {instance['images_iou_ge_0.9']} of {instance['images']} eval images reach IoU 0.9 or more, and the median of {instance['median_iou']:.3f} shows a typical image is near SAM's quality."),
              p(f"<b>Read the eval mean with care.</b> Eval beats dev ({m['iou', 'eval']:.3f} against {m['iou', 'dev']:.3f}), but this should not be read as generalising better. With 22 images from 4 scenes the interval is wide, and those scenes contain none of the white-on-white photos (bag, tray, towel) that cause most dev failures. The dev score is the safer guide to unseen scenes. Likewise, the 100% count accuracy on eval is limited by having no 3-egg images."),
              ]
    story.append(table([["Scene", "Images", "IoU", "Boundary F1", "Precision", "Recall"]] + [
        [s.scene.replace("_", " "), s.images, f"{s.iou:.3f}", f"{s.boundary_f1:.3f}", f"{s.precision:.3f}", f"{s.recall:.3f}"]
        for s in scenes.itertuples()], [width * 0.3] + [width * 0.14] * 5))
    story += [Spacer(1, 8), figure(os.path.join(EVAL, "plots.png")),
              p("3.2 What went wrong", h2),
              figure(os.path.join(EVAL, "worst.png"), width * 0.62),
              p("Green: agreement. Red: false positive. Blue: missed.", note)]
    w = {int(r.id): r for r in worst.itertuples()}
    story += bullets([
        f"<b>Submerged eggs (135, 145, 124).</b> Bubbles destroy the outline the method depends on. In 135 (IoU {w[135].iou:.2f}) the best ellipse is a wall tile, the only wrong object on eval. These three images also have the weakest ground truth.",
        f"<b>Shading splits (153, 28).</b> IoU {w[153].iou:.2f} and {w[28].iou:.2f}. Precision is near 1 but recall low: the ellipse stops at the terminator between light and shadow. GrabCut then likely treats the darker part as background, so refinement does not recover it.",
        "<b>Small eggs.</b> Every eval image with IoU below 0.9 has an egg under about 9% of the image, while all larger eggs score above 0.9. Small outlines give fewer and shorter arcs. This overlaps with the causes above, as the pot images have small eggs.",
        "<b>On dev: white on white.</b> Many of the worst dev images (3, 5, 109, 123, 125) are white eggs on white cloth or bags, where the chosen ellipse can be a bag flap or nothing. Eval has no such scene, so this failure is hidden from its score.",
        "<b>On dev: multi-egg images.</b> The 0.9 keep ratio that removes junk also drops real eggs, as in 140 and 144, where one of three eggs was found.",
    ])
    story += [p("3.3 Where to improve", h2)]
    story += bullets([
        "<b>Combine edge evidence with colour models.</b> A shadowed lower half fails GrabCut because it is darker. Normalising illumination first, for example dividing by a heavily blurred image, should reduce terminator cuts.",
        "<b>Use a shape prior during refinement.</b> Constraining GrabCut to stay near an ellipse would keep the whole egg in shading cases.",
        "<b>Replace the keep ratio with a learned threshold or a per-image model.</b> The same-size rule in T36 raised recall to 0.90, so a stricter version, such as also requiring a clean interior, may recover eggs without adding junk.",
        "<b>Evaluate on more scenes.</b> A larger eval set, or cross-validation by scene, would narrow the interval and cover white-on-white surfaces.",
    ])

    story += [p("4. Video tracking", h1),
              p("In the video, a hand carries each egg across the frame in turn, hiding it. Egg 1 goes right and comes back, then egg 2 goes right, so the eggs swap left-to-right order for a while and pass under the same hand. They look identical, so identity rests on motion and on reasoning about which egg is missing. I ran the final algorithm on each frame with three changes: a size range for this video, a lower keep ratio since two eggs are present, and no GrabCut, which is too slow per frame and unnecessary for tracking.")]
    story += bullets([
        "<b>False detections.</b> The first run had 6 IDs for 2 eggs, from table reflections and the title cards. Both are well-shaped but not white, so strong detections must now pass an absolute whiteness test.",
        "<b>Motion.</b> A Kalman filter with constant velocity predicts each egg. Once an egg is hidden for over 3 frames its prediction stops at the last seen position.",
        "<b>Association in four stages.</b> Tracks match to strong detections, then unmatched tracks match to nearby white slivers, such as an egg showing between fingers. Remaining strong detections go to lost tracks, nearest first, and finally lost tracks can claim large white blobs.",
        "<b>New IDs.</b> Only made when no lost track can claim a detection that persists for 8 frames, because eggs do not appear from nowhere.",
    ])
    story.append(p("Result", h2))
    story.append(table([["Measure", "Value"], ["Unique egg IDs in the video", len(tracking["track_ids"])],
                        ["Keyframes checked, by egg", total], ["Correct identity", tracking["correct"]],
                        ["Wrong identity", tracking["wrong_id"]], ["No track, with the egg hidden in the hand", tracking["no_track"]]],
                       [width * 0.75, width * 0.25]))
    story += [Spacer(1, 6), p("I labelled positions by hand on 32 keyframes from frames 470 to 995, covering all of the carries, and viewed frames 726 to 765 at three-frame steps, where both eggs are under the hand, to confirm the identities. Labels come from thumbnails, so position agreement uses a tolerance of 50 px."),
              p("<b>Limits.</b> This is a two-egg clip, and the reassignment rule relies on it: with several eggs hidden at once, identity would be a guess based on last position. The masks are ellipses on this video and may run slightly off while an egg is spinning. The whiteness threshold is tuned to this video and would need adjusting for others."),
              figure(os.path.join(OUT, "tracking_sheet.jpg"), width * 0.72),
              p("Every 15th frame from 470 to 995. Egg 1 goes right and returns by frame 695, then egg 2 goes right from frame 755.", note)]

    story += [p("5. Reproducing this work", h1), p("Setup and run instructions are in the repository's README. Results here come from <font name='Courier'>outputs/</font>, regenerated by <font name='Courier'>src/build_report.py</font>.")]

    os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
    SimpleDocTemplate(os.path.join(ROOT, "report", "report.pdf"), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=17 * mm, bottomMargin=17 * mm, title="CV Intern Task: Egg Segmentation and Tracking").build(story)
    print("wrote report/report.pdf")


if __name__ == "__main__":
    main()
