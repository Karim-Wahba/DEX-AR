#!/bin/bash
#SBATCH --job-name=voc-full
#SBATCH --partition=a100-galvani          # cluster-specific: change for yours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Run #55 -- the FULL PascalVOC 2012 segmentation val split (1449 images, not the
# 150-image subset every prior VOC number in this project used), one (model, arm)
# per job so the four models run in parallel.
#
# Two arms per model, because the rank:2 number is only readable against the
# as-published one measured on the same images:
#   ARM=baseline   DEX-AR exactly as published
#   ARM=rank2      MAP_FILTER=rank:2, delta^t KEPT (run #54 point 4)
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-VL-8B-Instruct,ENV=dexar,IMG_SIZE=native,ARM=rank2 \
#          slurm/voc_full.sbatch
#
# ENV: `dexar` (transformers 4.57.6) for Qwen3-VL and OneVision-1.5; `dexar-tf5`
# (5.15.1) for GLM-4.6V and OneVision-2. See slurm/ov15_smoke.sbatch for why
# OneVision-1.5 cannot use tf5.

set -u
# Slurm runs a COPY of this script, so $0 is not the repo. Submit from the
# repo root, or set DEXAR_REPO.
cd "${DEXAR_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

# Paths come from paths.ini / paths.local.ini (see config.py). Exporting
# VOC_ROOT here is optional and only overrides them for this job.
ENV="${ENV:-dexar}"
PY="${PY:-$HOME/.conda/envs/$ENV/bin/python}"
IMG_SIZE="${IMG_SIZE:-native}"
ARM="${ARM:-rank2}"
LIMIT="${LIMIT:-0}"          # 0 = the whole 1449-image val split
case "$ARM" in
  baseline) FILTER="" ;;
  rank2)    FILTER="rank:2" ;;
  rank1)    FILTER="rank:1" ;;
  *) echo "unknown ARM=$ARM"; exit 2 ;;
esac

echo "=== run #55 full-VOC | job ${SLURM_JOB_ID:-none} | $(date -Is)"
echo "  model  : $MODEL"
echo "  env    : $ENV -> $($PY -c 'import transformers;print(transformers.__version__)')"
echo "  size   : $IMG_SIZE   arm: $ARM   map_filter: '${FILTER:-none}'   limit: $LIMIT"

# Smoke first: a 2-image run into scratch. Four load-path faults were found the
# hard way in #50, and a 4-hour job is not the place to find the fifth.
SMOKE="/tmp/voc-full-smoke-${SLURM_JOB_ID:-$$}"
MODEL="$MODEL" VARIANT=baseline IMG_SIZE="$IMG_SIZE" LAYER_INDEX=0 LIMIT=2 \
  MAP_FILTER="$FILTER" DROP_DELTA=0 RESULTS_DIR="$SMOKE" \
  "$PY" experiments/eval_masks.py || { echo "SMOKE FAILED"; rm -rf "$SMOKE"; exit 1; }
rm -rf "$SMOKE"

echo; echo "=== full split starting $(date -Is)"
MODEL="$MODEL" VARIANT=baseline IMG_SIZE="$IMG_SIZE" LAYER_INDEX=0 LIMIT="$LIMIT" \
  MAP_FILTER="$FILTER" DROP_DELTA=0 TAG_SUFFIX="${TAG_SUFFIX:-_full1449}" \
  "$PY" experiments/eval_masks.py
rc=$?
echo "=== finished rc=$rc $(date -Is)"
exit $rc
