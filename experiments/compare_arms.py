"""Paired comparison of eval_masks.py arms against a reference run.

Every arm scores the SAME 150 PascalVOC images, so the informative statistic is
the per-image paired difference, not the gap between two means. A mean gap of
+2.8 IoU could be 150 images each moving +2.8, or 10 images moving +42 and 140
unchanged; those license very different claims and the means cannot tell them
apart.

Also reports corr(GT area, per-image gain) and the gain split at the median
area. This separates a real effect from a metric artifact: soft-IoU and EPG both
rise with target area (run #41 measured corr(area, EPG) = +0.97), so an
intervention that merely spreads map mass must show a gain that GROWS with area.
A gain that is flat in area cannot be explained that way.

Reports, per metric: the paired mean difference, a bootstrap 95% CI over images,
a paired sign-flip permutation p (20,000 flips -- no scipy in the `dexar` env,
and that env is not to be modified), and the win rate. Nulls from run #41 are printed on
the same scale so "better than baseline" and "better than noise" stay visibly
separate questions -- on this benchmark they are not the same question.

  python experiments/compare_arms.py <ref_dir> <arm_dir> [<arm_dir> ...]
"""
import json
import os
import sys

import numpy as np

METRICS = ("soft_iou", "iou", "epg", "snr")
NULLS = {"soft_iou": 19.17, "iou": 26.58, "epg": 26.37, "snr": 0.00}
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402  -- every path comes from here; see paths.ini


def load(d):
    if not os.path.isabs(d):
        d = os.path.join(config.RESULTS_ROOT, d)
    rows = json.load(open(os.path.join(d, "per_image.json")))
    return {r["id"]: r for r in rows}


def signflip_p(d, n=20000, seed=0):
    """Two-sided paired permutation test: under H0 each image's difference is
    equally likely to carry either sign, so flipping signs at random generates
    the null distribution of the mean. Ties contribute 0 either way and are
    kept, which is what makes a true null arm report p near 1 rather than
    silently dropping to a smaller effective n."""
    if not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return (np.sum(np.abs(null) >= abs(d.mean())) + 1) / (n + 1)


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ref_dir, arm_dirs = sys.argv[1], sys.argv[2:]
    ref = load(ref_dir)
    print(f"reference: {os.path.basename(ref_dir.rstrip('/'))}  (n={len(ref)})\n")

    for arm_dir in arm_dirs:
        arm = load(arm_dir)
        ids = sorted(set(ref) & set(arm))
        if len(ids) != len(ref) or len(ids) != len(arm):
            print(f"  NOTE: {len(ids)} shared ids "
                  f"(ref {len(ref)}, arm {len(arm)}) -- comparing the overlap")
        print(f"=== {os.path.basename(arm_dir.rstrip('/'))}   n={len(ids)}")
        area = np.array([ref[i]["area"] for i in ids], float)
        med_area = np.median(area)
        print(f"  {'metric':9s} {'ref':>7s} {'arm':>7s} {'paired d':>9s} "
              f"{'95% CI':>17s} {'perm p':>11s} {'win%':>6s} {'vs null':>9s}"
              f" | {'r(area,d)':>9s} {'smallGT':>8s} {'largeGT':>8s}")
        for m in METRICS:
            a = np.array([ref[i][m] for i in ids], float)
            b = np.array([arm[i][m] for i in ids], float)
            # SNR is undefined when a map puts no mass outside the mask (one
            # such image in the 150). Drop the pair rather than the image, so
            # every other metric still uses the full n -- and say so.
            ok = np.isfinite(a) & np.isfinite(b)
            d_all = b - a
            a, b = a[ok], b[ok]
            d = b - a
            ar = area[ok]
            r_area = (np.corrcoef(ar, d)[0, 1] if np.std(d) > 0 else 0.0)
            small = d[ar < med_area].mean() if (ar < med_area).any() else np.nan
            large = d[ar >= med_area].mean() if (ar >= med_area).any() else np.nan
            lo, hi = boot_ci(d)
            p = signflip_p(d)
            win = 100.0 * np.mean(d > 0)
            drop = "" if ok.all() else f"  (n={ok.sum()})"
            print(f"  {m:9s} {a.mean():7.2f} {b.mean():7.2f} {d.mean():+9.2f} "
                  f"[{lo:+6.2f},{hi:+6.2f}] {p:11.2e} {win:5.1f}% "
                  f"{b.mean() - NULLS[m]:+9.2f}{drop}"
                  f" | {r_area:+9.3f} {small:+8.2f} {large:+8.2f}")
        print()


if __name__ == "__main__":
    main()
