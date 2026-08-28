"""Reproduce DEX-AR's PascalVOC segmentation numbers, then measure Qwen3-VL.

THE POINT OF THIS SCRIPT
------------------------
A reimplementation bug is a live explanation for any surprising result, so this
runs the paper's own protocol on the paper's own dataset and model, and either
reproduces its table or does not. Reproduce the LLaVA-1.5 row FIRST; only then
is a number measured on another model worth reading.

Target (paper Table 1, LLaVA-1.5-7B on PascalVOC):

    soft-IoU 17.70    IoU 36.34    EPG 27.75

If those come out, the implementation is trustworthy and DEX-AR's IoU on
Qwen3-VL becomes meaningful -- and decides whether the artifacts found in runs
#22-#32 cost localisation or are cosmetic. If they do not come out, stop and fix
the implementation before reading anything else.

PROTOCOL (from the paper, recorded in wiki/entities/bousselham-dexar-2026.md)
  prompt        "USER: <image>\\nClassify the image. ASSISTANT:" -- and the model
                FREE-RUNS. Segmentation is not teacher-forced; the sentence map
                (Eq. 6) is built over the model's own generated tokens.
  ground truth  the union of the masks of ALL objects present, i.e. every
                non-background non-void pixel of SegmentationClass. This rewards
                diffuse maps, which is why a spiky artifact can survive it.
  layers        all L layers (paper Eq. 5 / Appendix Eq. 26). Note the released
                code defaults to the last 10; LAYER_INDEX exposes both.
  soft-IoU      SAM / (SA + SM - SAM), threshold-free
  IoU           thresholded, threshold chosen per prediction by maximising IoU
                over k = 20 equally spaced values
  EPG           SAM / SA x 100
  SNR           10 log( <E,M>/|M|_1  /  <E,1-M>/|1-M|_1 )
  All reported x100, as the paper does.

The attribution map is upsampled bilinearly to the mask resolution rather than
the mask being pooled to the token grid -- the standard CAM protocol, and the
only choice that keeps IoU thresholds comparable with published values.

Usage:
  MODEL=llava-hf/llava-1.5-7b-hf IMG_SIZE=336 LIMIT=200 \\
    ~/.conda/envs/dexar/bin/python experiments/eval_masks.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402  -- every path comes from here; see paths.ini
from dexar import DexarWrapper  # noqa: E402

MODEL = os.environ.get("MODEL", "llava-hf/llava-1.5-7b-hf")
VARIANT = os.environ.get("VARIANT", "baseline")   # free-text tag for the run dir
# Rank prefilter: MAP_FILTER=rank:2 applies a 3x3 k-th order-statistic filter to
# every (token, layer, head) map before Eq. 5 scores it. DROP_DELTA=1 sets the
# Eq. 6 token weights to 1 -- delta^t is a second max-vs-max comparison that any
# max-lowering prefilter kills. Both off by default = DEX-AR as published.
MAP_FILTER = os.environ.get("MAP_FILTER", "").strip() or None
DROP_DELTA = os.environ.get("DROP_DELTA", "0") not in ("", "0")
_SIZE_RAW = os.environ.get("IMG_SIZE", "336")
# IMG_SIZE=native skips the square resize and lets the processor's smart-resize
# pick the grid -- the only way to run Qwen3-VL at its designed native dynamic
# resolution. A forced square is in-distribution for LLaVA (which always squashes
# to 336) and out-of-distribution for Qwen3-VL, so the two are NOT comparable
# unless this is run both ways.
NATIVE = _SIZE_RAW.lower() == "native"
SIZE = None if NATIVE else int(_SIZE_RAW)
LAYER_INDEX = os.environ.get("LAYER_INDEX", "0")   # 0 = all layers, the paper
TEXT_SET = os.environ.get("TEXT_SET", "all_non_image")
LIMIT = int(os.environ.get("LIMIT", "0"))          # 0 = the whole val split
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "20"))
N_THRESH = int(os.environ.get("N_THRESH", "20"))   # the paper's k = 20
MAP = os.environ.get("MAP", "sentence")            # sentence | sentence_unfiltered
RESULTS_ROOT = os.environ.get("RESULTS_DIR") or config.RESULTS_ROOT
# Appended to the results directory name. The tag encodes model/variant/size/
# layer but NOT the image count, so a full-split run would otherwise overwrite
# the 150-image run of the same arm. Run #55 passes TAG_SUFFIX=_full1449.
TAG_SUFFIX = os.environ.get("TAG_SUFFIX", "")
# VOCdevkit/VOC2012, from paths.ini / paths.local.ini / $VOC_ROOT. Resolved by
# voc_root() at first use rather than at import, so a missing path fails where
# it is needed with the fix in the message, not as an import-time traceback.
VOC = None
# The paper's classification query. LLaVA needs the full template carrying the
# '<image>' placeholder; Qwen3-VL takes the bare instruction and wraps it in its
# own chat template. Override wholesale with PROMPT.
INSTRUCTION = os.environ.get("INSTRUCTION", "Classify the image.")


def build_prompt():
    if os.environ.get("PROMPT"):
        return os.environ["PROMPT"]
    if "llava" in MODEL.lower() or "bakllava" in MODEL.lower():
        return f"USER: <image>\n{INSTRUCTION} ASSISTANT:"
    return INSTRUCTION


PROMPT = build_prompt()


def voc_root():
    global VOC
    if VOC is None:
        VOC = config.require_voc_root()
    return VOC


def load_val():
    ids = open(f"{voc_root()}/ImageSets/Segmentation/val.txt").read().split()
    return ids[:LIMIT] if LIMIT else ids


def gt_union(seg_png):
    """Union of every object present: non-background (0) and non-void (255)."""
    a = np.array(Image.open(seg_png))
    return ((a > 0) & (a != 255)).astype(np.float64)


def upsample(heat, hw):
    t = torch.as_tensor(np.ascontiguousarray(heat), dtype=torch.float32)[None, None]
    up = F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()
    return (up - up.min()) / (up.max() - up.min()) if up.max() > up.min() \
        else np.zeros_like(up)


def score(a, m):
    """The paper's four spatial metrics, all x100."""
    sam, sa, sm = float((a * m).sum()), float(a.sum()), float(m.sum())
    soft = sam / max(sa + sm - sam, 1e-12)
    epg = sam / max(sa, 1e-12)

    mb = m > 0.5
    best = 0.0
    for t in np.linspace(0.0, 1.0, N_THRESH, endpoint=False):
        pb = a > t
        u = float((pb | mb).sum())
        if u > 0:
            best = max(best, float((pb & mb).sum()) / u)

    inv = 1.0 - m
    num = sam / max(sm, 1e-12)
    den = float((a * inv).sum()) / max(float(inv.sum()), 1e-12)
    snr = 10.0 * np.log10(num / den) if den > 0 and num > 0 else float("nan")
    return dict(soft_iou=100 * soft, iou=100 * best, epg=100 * epg, snr=snr,
                area=100 * float(m.mean()))


def main():
    li = int(LAYER_INDEX) if LAYER_INDEX.lstrip("-").isdigit() else \
        [int(x) for x in LAYER_INDEX.strip("[]").split(",")]
    extra_tag = ((f"_{MAP_FILTER.replace(':', '')}" if MAP_FILTER else "")
                 + ("_nodelta" if DROP_DELTA else ""))
    tag = (f"{MODEL.split('/')[-1]}_eval-masks_voc_{VARIANT.replace(':', '-')}{extra_tag}_"
           f"{'native' if NATIVE else str(SIZE) + 'px'}"
           f"_L{str(LAYER_INDEX).strip('[]').replace(':', '-')}_{MAP}{TAG_SUFFIX}")
    run_dir = os.path.join(RESULTS_ROOT, tag)
    os.makedirs(run_dir, exist_ok=True)

    ids = load_val()
    w = DexarWrapper.from_pretrained(MODEL, device="cuda", layer_index=li,
                                     text_set=TEXT_SET)

    hdr = [f"run: {tag}", f"model: {MODEL}", f"variant: {VARIANT}",
           f"map_filter: {MAP_FILTER}  drop_delta: {DROP_DELTA}",
           f"dataset: PascalVOC 2012 segmentation val, {len(ids)} images",
           f"prompt: {PROMPT!r} (FREE-RUN, max {MAX_NEW} tokens -- not teacher-forced)",
           f"ground truth: union of all objects present",
           f"layers: {w.layers[0]}..{w.layers[-1]} ({len(w.layers)} of {w.num_layers})",
           f"image size: {'NATIVE (processor smart-resize)' if NATIVE else SIZE}"
           f"  text_set: {TEXT_SET}  IoU thresholds: {N_THRESH}",
           f"map: {MAP}",
           "",
           "TARGET (paper Table 1, LLaVA-1.5-7B): soft-IoU 17.70  IoU 36.34  EPG 27.75",
           ""]
    print("\n".join(hdr), flush=True)

    rows, fails = [], 0
    for i, iid in enumerate(ids, 1):
        voc = voc_root()
        jpg, png = f"{voc}/JPEGImages/{iid}.jpg", f"{voc}/SegmentationClass/{iid}.png"
        try:
            im = Image.open(jpg).convert("RGB")
            m = gt_union(png)
            res = w.compute_dexar(
                im if NATIVE else im.resize((SIZE, SIZE)), target_sentence=None,
                prompt=PROMPT, max_new_tokens=MAX_NEW,
                map_filter=MAP_FILTER, drop_delta=DROP_DELTA)
        except Exception as exc:                                   # noqa: BLE001
            fails += 1
            print(f"  [{i}/{len(ids)}] {iid} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            continue

        heat = getattr(res, "sentence_heatmap" if MAP == "sentence"
                       else "sentence_heatmap_unfiltered")
        hnp = heat.detach().float().cpu().numpy()
        a = upsample(hnp, m.shape)
        # A degenerate (constant / all-zero) map upsamples to zeros and scores
        # soft-IoU 0 with SNR nan. It is a real failure, not a dropped image, so
        # it stays in the mean -- but it is counted, because five of them in 150
        # is a different result from none.
        row = dict(id=iid, n_tokens=len(res.tokens),
                   degenerate=bool(hnp.max() <= hnp.min()),
                   caption=" ".join(res.tokens)[:80])
        row.update(score(a, m))
        rows.append(row)
        if i % 25 == 0 or i <= 5:
            run = {k: float(np.mean([r[k] for r in rows]))
                   for k in ("soft_iou", "iou", "epg")}
            print(f"  [{i}/{len(ids)}] {iid}  running mean: "
                  f"soft-IoU {run['soft_iou']:.2f}  IoU {run['iou']:.2f}  "
                  f"EPG {run['epg']:.2f}", flush=True)
        # Checkpoint, so a long run is readable (and survivable) mid-flight.
        if i % 50 == 0:
            with open(os.path.join(run_dir, "per_image.json"), "w") as f:
                json.dump(rows, f, indent=1)

    out = list(hdr)
    out.append(f"--- results over {len(rows)} images ({fails} failed) ---")
    out.append(f"  mean object area: {np.mean([r['area'] for r in rows]):.1f}% of the image")
    n_deg = sum(1 for r in rows if r.get("degenerate"))
    out.append(f"  degenerate (constant) maps: {n_deg} of {len(rows)}")
    out.append("")
    out.append(f"  {'metric':<10} {'mean':>8} {'std':>8}   {'paper':>8}   delta")
    paper = {"soft_iou": 17.70, "iou": 36.34, "epg": 27.75}
    for k in ("soft_iou", "iou", "epg", "snr"):
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[~np.isnan(v)]
        p = paper.get(k)
        tail = f"   {p:8.2f}   {np.mean(v) - p:+.2f}" if p else "          --"
        out.append(f"  {k:<10} {np.mean(v):8.2f} {np.std(v):8.2f}{tail}")
    if paper and rows:
        ok = all(abs(np.mean([r[k] for r in rows]) - paper[k]) < 2.0 for k in paper)
        out.append("")
        out.append(f"  REPRODUCED (all three within 2.0 absolute): {ok}")

    text = "\n".join(out)
    with open(os.path.join(run_dir, "metrics.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(run_dir, "per_image.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print("\n" + "\n".join(out[len(hdr):]), flush=True)
    print("\nsaved ->", run_dir, flush=True)


if __name__ == "__main__":
    main()
