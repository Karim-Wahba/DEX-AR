"""Chance lines for the full PascalVOC val split (run #55).

Run #41 measured the nulls on the 150-image subset (random smooth map soft-IoU
19.17 / IoU 26.58 / EPG 26.37 / SNR 0.00). Run #55 scores 1449 images, and every
one of these metrics is a function of object area, so the subset's numbers are
not the split's numbers. This recomputes them through `eval_masks.score` and
`eval_masks.upsample` -- the same code the model runs go through -- so the table
compares like with like.

Two nulls, both from run #41:
  random smooth  i.i.d. uniform noise on the token grid, upsampled bilinearly
                 to mask resolution and min-maxed. This is what "a map with no
                 information but the right smoothness" scores.
  uniform        a constant map. Scores EPG = mean object area by construction,
                 and beats every method in the paper's Table 1 on soft-IoU.

  SEEDS=5 GRID=24 ~/.conda/envs/dexar/bin/python experiments/voc_nulls.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.eval_masks import gt_union, load_val, score, upsample, voc_root  # noqa: E402

SEEDS = int(os.environ.get("SEEDS", "5"))
GRID = int(os.environ.get("GRID", "24"))   # 336px/14 -- LLaVA-1.5's token grid


def main():
    ids = load_val()
    print(f"nulls over {len(ids)} VOC val images, grid {GRID}x{GRID}, "
          f"{SEEDS} seeds", flush=True)
    rand, unif = [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(s)
        rows_r, rows_u = [], []
        for iid in ids:
            m = gt_union(f"{voc_root()}/SegmentationClass/{iid}.png")
            rows_r.append(score(upsample(rng.random((GRID, GRID)), m.shape), m))
            rows_u.append(score(np.ones(m.shape), m))
        rand.append(rows_r)
        unif.append(rows_u)
        print(f"  seed {s} done", flush=True)

    for name, seeds in (("random smooth", rand), ("uniform", unif)):
        print(f"\n{name}:")
        for k in ("soft_iou", "iou", "epg", "snr"):
            per_seed = np.array([np.nanmean([r[k] for r in rows]) for rows in seeds])
            print(f"  {k:<9} {per_seed.mean():7.2f}  (sd over seeds {per_seed.std():.3f})")


if __name__ == "__main__":
    main()
