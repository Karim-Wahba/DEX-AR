"""Every filesystem path this repo needs, resolved in one place.

Scripts import this instead of hardcoding a path or growing their own default,
so moving the data or the results directory is a one-line edit in `paths.ini`
rather than a grep across `experiments/`.

Precedence, highest first:

  1. an environment variable   -- VOC_ROOT=/other/voc python experiments/...
     for one-off runs, and how slurm/voc_full.sbatch passes overrides through
  2. paths.local.ini           -- your machine's paths; gitignored, so it never
     travels with a commit or collides with someone else's checkout
  3. paths.ini                 -- the tracked template, with placeholders
  4. the built-in default      -- only where one is meaningful (results/, logs/)

`~` is expanded everywhere. Relative paths resolve against the repo root, so
`results_root = results` and an absolute path both work.

  python -m config     # print what resolved, and where each value came from
"""

import configparser
import os

# The repo root is this file's directory. DEXAR_REPO overrides it, which only
# matters if config.py is imported from a copy living outside the checkout.
REPO = os.environ.get("DEXAR_REPO") or os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.expanduser(REPO))

_INI = os.path.join(REPO, "paths.ini")
_LOCAL_INI = os.path.join(REPO, "paths.local.ini")
_cfg = configparser.ConfigParser()
_cfg.read([_INI, _LOCAL_INI])            # later file wins


def _resolve(key, default=""):
    """Return (value, source) for one path key."""
    env = os.environ.get(key.upper())
    if env:
        return _abs(env), f"env {key.upper()}"
    for path, name in ((_LOCAL_INI, "paths.local.ini"), (_INI, "paths.ini")):
        if os.path.exists(path):
            local = configparser.ConfigParser()
            local.read(path)
            v = local.get("paths", key, fallback="").strip()
            if v and not v.startswith("<"):      # "<set me>" placeholders don't count
                return _abs(v), name
    return (_abs(default) if default else ""), "default"


def _abs(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def get(key, default=""):
    return _resolve(key, default)[0]


def require(key, what):
    """A path with no sensible default. Fails with the fix, not a stack trace."""
    v = get(key)
    if not v:
        raise SystemExit(
            f"{key} is not set. It must point at {what}.\n"
            f"  Set it in {os.path.relpath(_LOCAL_INI, os.getcwd())} under [paths],\n"
            f"  or pass {key.upper()}=... in the environment.")
    return v


#: PascalVOC 2012 devkit root -- the directory holding ImageSets/, JPEGImages/
#: and SegmentationClass/. No default: it is data, not part of the checkout.
#: Read it through require_voc_root(), so the error arrives at use, not import.
def require_voc_root():
    voc = require("voc_root", "the VOCdevkit/VOC2012 directory")
    val = os.path.join(voc, "ImageSets", "Segmentation", "val.txt")
    if not os.path.exists(val):
        raise SystemExit(f"no {val}\n  -- is voc_root={voc!r} really the VOC2012 root?")
    return voc


RESULTS_ROOT = get("results_root", "results")   # eval output: metrics + per-image
LOGS_ROOT = get("logs_root", "logs")            # slurm job output
BASELINES_ROOT = get("baselines_root", "baselines")   # cached activation tensors
HF_CACHE = get("hf_cache")                      # optional: HuggingFace model cache


def _main():
    print(f"REPO           {REPO}")
    print(f"  paths.ini       {'found' if os.path.exists(_INI) else 'MISSING'}")
    print(f"  paths.local.ini {'found' if os.path.exists(_LOCAL_INI) else 'not present'}")
    print()
    for key, default in (("voc_root", ""), ("results_root", "results"),
                         ("logs_root", "logs"), ("baselines_root", "baselines"),
                         ("hf_cache", "")):
        value, source = _resolve(key, default)
        print(f"  {key:<15} {value or '(unset)':<60} [{source}]")


if __name__ == "__main__":
    _main()
