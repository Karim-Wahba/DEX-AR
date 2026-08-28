"""Assemble run #55's cross-model table in the DEX-AR paper's own format.

The paper's Table 1 reports soft-IoU, IoU and EPG per model on PascalVOC. This
prints those three plus SNR (which is the only one of the four with a defined
zero: a uniform map scores 0.00), the image count, and the number of degenerate
maps -- a constant map scores soft-IoU 0 and drags a mean without being a
localisation failure of the kind the other columns describe.

Every row is the FULL 1449-image val split. Quote nothing here against run #41's
nulls: those were measured on the 150-image subset. `experiments/voc_nulls.py`
recomputes them on this split.

  python experiments/voc_table.py                    # all _full1449 runs found
  python experiments/voc_table.py <dir> [<dir> ...]  # explicit
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402  -- every path comes from here; see paths.ini
METRICS = ("soft_iou", "iou", "epg", "snr")
PAPER = {  # DEX-AR Table 1, PascalVOC
    "llava-1.5-7b": (17.70, 36.34, 27.75),
}


def label(d):
    b = os.path.basename(d.rstrip("/"))
    b = b.replace("_eval-masks_voc_", " | ").replace("_sentence_full1449", "")
    return b.replace("_full1449", "")


def row(d):
    rows = json.load(open(os.path.join(d, "per_image.json")))
    out = {"n": len(rows),
           "deg": sum(1 for r in rows if r.get("degenerate")),
           "area": float(np.mean([r["area"] for r in rows]))}
    for m in METRICS:
        v = np.array([r[m] for r in rows], float)
        v = v[np.isfinite(v)]
        out[m] = float(v.mean())
        out[m + "_n"] = int(v.size)
    return out


def main():
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(config.RESULTS_ROOT, "*_full1449")))
    if not dirs:
        sys.exit("no *_full1449 result directories found")
    w = max(len(label(d)) for d in dirs)
    print(f"{'run':<{w}} {'n':>5} {'soft-IoU':>9} {'IoU':>7} {'EPG':>7} "
          f"{'SNR':>7} {'deg':>5} {'area%':>7}")
    print("-" * (w + 50))
    for d in dirs:
        if not os.path.exists(os.path.join(d, "per_image.json")):
            print(f"{label(d):<{w}}   (no per_image.json -- still running?)")
            continue
        r = row(d)
        snr = f"{r['snr']:+7.2f}" if r["snr_n"] else "     --"
        print(f"{label(d):<{w}} {r['n']:>5} {r['soft_iou']:>9.2f} {r['iou']:>7.2f} "
              f"{r['epg']:>7.2f} {snr} {r['deg']:>5} {r['area']:>7.1f}")
    print()
    for k, (s, i, e) in PAPER.items():
        print(f"paper (DEX-AR Table 1, {k}, 1449 images): "
              f"soft-IoU {s:.2f}  IoU {i:.2f}  EPG {e:.2f}")


if __name__ == "__main__":
    main()
