# DEX-AR — full-PascalVOC evaluation and the rank prefilter

A fork of [WalBouss/DEX-AR](https://github.com/WalBouss/DEX-AR) that adds:

- a **reproduction harness** for the paper's PascalVOC segmentation protocol, run
  over the whole 1449-image val split rather than a subset;
- **four more architectures** — Qwen3-VL, GLM-4.6V-Flash, LLaVA-OneVision-1.5 and
  LLaVA-OneVision-2 — alongside the original LLaVA-1.5 and BakLLaVA;
- **one optional change to Eq. 5**, a 3x3 rank prefilter on each head map, off by
  default;
- **chance baselines**, because soft-IoU and EPG both track object area and
  neither separates signal from noise on its own.

## What DEX-AR computes

DEX-AR explains an autoregressive VLM by differentiating intermediate-layer
logits with respect to the attention probabilities over visual tokens, then
filtering twice: per head, by how much more a head attends to image than to text
(Eq. 5), and per generated token, by the same difference taken over global maxima
(Eq. 6).

**[`docs/equations/dexar_equations.pdf`](docs/equations/dexar_equations.pdf)** is
the reference: every equation as published, the prefilter stated as a
modification of exactly two of them, the metric definitions, and a table mapping
each equation to the function that implements it. Source `.tex` is alongside it.

## The rank prefilter

Eq. 5 scores an attention head by its single tallest image cell, so a head whose
map is one isolated spike scores as highly as a head that has found the object.
The prefilter replaces each per-(token, layer, head) image map by the **k-th
order statistic of each cell's 3x3 neighbourhood** before Eq. 5 sees it — an
operator an isolated cell cannot survive but a coherent region can. `S_img` is
then rescaled per (token, layer) so the Eq. 5 subtraction does not compare a
filtered quantity against an unfiltered `S_text`. Eq. 5 and Eq. 6 are otherwise
unmodified.

```python
out = wrapper.compute_dexar(image, prompt=prompt, map_filter="rank:2")
```

`rank:0` is erosion, `rank:4` the median, `rank:8` dilation. `rank:1` and
`rank:2` are the useful settings; above the median the maps collapse to chance.
`rank:k:noscale` ablates the rescale. **With no `map_filter` this is DEX-AR
exactly as published.**

## Where the explanations live

| document | what it covers |
|---|---|
| [`docs/equations/dexar_equations.pdf`](docs/equations/dexar_equations.pdf) | the equations before and after, the metrics, and the equation-to-code map |
| [`envs/README.md`](envs/README.md) | the two conda environments, and why each model is pinned to the transformers version it is |
| `paths.ini` | every filesystem path; `config.py` resolves them |

## Steps to run

### 1. Environments

Two conda envs, because no single transformers version runs all five models —
see [`envs/README.md`](envs/README.md) for which model needs which and why.

```bash
conda env create -f envs/dexar.yml          # transformers 4.57.6
conda env create -f envs/dexar-tf5.yml      # transformers 5.15.1

conda activate dexar     && pip install -e .
conda activate dexar-tf5 && pip install -e .
```

| env | transformers | models |
|---|---|---|
| `dexar` | 4.57.6 | LLaVA-1.5, BakLLaVA, Qwen3-VL, LLaVA-OneVision-1.5 |
| `dexar-tf5` | 5.15.1 | GLM-4.6V-Flash, LLaVA-OneVision-2 |

### 2. Configure paths, once

Every path lives in one file. Nothing is hardcoded anywhere in the repo.

```bash
cp paths.ini paths.local.ini      # gitignored, so it never travels with a commit
$EDITOR paths.local.ini           # set voc_root
python -m config                  # print what resolved, and where each came from
```

| key | what | default |
|---|---|---|
| `voc_root` | VOCdevkit/VOC2012, holding `ImageSets/`, `JPEGImages/`, `SegmentationClass/` | none — required |
| `results_root` | where runs write `metrics.txt` and `per_image.json` | `results/` |
| `logs_root` | Slurm job output | `logs/` |
| `baselines_root` | cached activation tensors | `baselines/` |
| `hf_cache` | HuggingFace model cache | HF default |

Precedence is environment variable > `paths.local.ini` > `paths.ini` > built-in
default, so `VOC_ROOT=/other/voc python experiments/eval_masks.py` still works
for a one-off without editing anything. Relative paths resolve against the repo
root; `~` is expanded. `python -m config` names the source of each value, which
is what you want when a run writes somewhere unexpected.

### 3. Smoke-test on two images

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct IMG_SIZE=native LIMIT=2 MAP_FILTER=rank:2 \
  python experiments/eval_masks.py
```

Paths come from the config; only the run's own knobs are passed here.

### 4. One full evaluation

```bash
MODEL=Qwen/Qwen3-VL-8B-Instruct \
IMG_SIZE=native \        # or a pixel count, e.g. 336 for LLaVA-1.5
LAYER_INDEX=0 \          # 0 = all layers, the paper's Eq. 5
MAP_FILTER=rank:2 \      # omit for DEX-AR as published
LIMIT=0 \                # 0 = the whole 1449-image val split
TAG_SUFFIX=_full1449 \
  python experiments/eval_masks.py
```

Results land in `<results_root>/<tag>/` as `metrics.txt` and `per_image.json`.
The protocol is the paper's: the model **free-runs** from "Classify the image."
(segmentation is not teacher-forced), ground truth is the union of every object
present, soft-IoU is threshold-free, IoU takes the best of 20 thresholds, EPG is
`SAM/SA`. All metrics x100. Definitions are in the PDF.

Other knobs: `DROP_DELTA=1` sets the Eq. 6 token weights to 1 (keep it off —
dropping them costs SNR), `TEXT_SET` selects the `S_text` token set
(`all_non_image` is the published one), `MAX_NEW_TOKENS`, `N_THRESH`, `PROMPT`,
`RESULTS_DIR`.

### 5. The full sweep, on Slurm

One job per (model, arm); each smoke-tests on 2 images before committing to the
split. Paths come from the config, so nothing needs exporting. Edit the
`#SBATCH` partition line for your cluster.

```bash
mkdir -p logs
for spec in \
  "Qwen/Qwen3-VL-8B-Instruct                      dexar     native qwen3vl" \
  "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct       dexar     native ov15"    \
  "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct dexar-tf5 native ov2"     \
  "zai-org/GLM-4.6V-Flash                         dexar-tf5 native glm"     ; do
  set -- $spec
  for ARM in baseline rank2; do
    sbatch --job-name="voc-$4-$ARM" \
      --export=ALL,MODEL=$1,ENV=$2,IMG_SIZE=$3,ARM=$ARM slurm/voc_full.sbatch
  done
done
```

`ARM=baseline` is DEX-AR as published; `ARM=rank2` / `ARM=rank1` set the
prefilter. Run both — a prefilter number is only readable against the
as-published number **on the same images**, since every one of these metrics
moves with object area.

### 6. Read the results

```bash
python experiments/voc_nulls.py    # chance lines for this split (CPU, ~10 min)
python experiments/voc_table.py    # the cross-model table
python experiments/compare_arms.py <baseline_dir> <arm_dir>   # paired stats
```

`voc_nulls.py` matters. On the 1449-image split a **random smooth map** scores
soft-IoU 18.41 / IoU 25.46 / EPG 25.24 / SNR -0.01, and a **uniform map** scores
25.24 on all three of the paper's metrics. Soft-IoU and EPG both rise with object
area, so neither separates signal from noise on its own; quote every number
against these lines, and prefer SNR, the only one of the four with a defined
zero. `compare_arms.py` reports per-image paired differences, a bootstrap CI, a
sign-flip permutation p, and the gain split by object area — an intervention that
merely spreads map mass shows a gain that grows with area.


## Usage as a library

```python
from PIL import Image
from dexar import DexarWrapper, visualize

model = DexarWrapper.from_pretrained("llava-hf/llava-1.5-7b-hf", device="cuda")
image = Image.open("./assets/cat_and_dog.jpg").convert("RGB").resize((336, 336))

result = model.compute_dexar(
    image=image,
    target_sentence="The image features a dog and a cat sitting together in a grassy field",
    prompt="USER: <image>\nDescribe the image. ASSISTANT:",
    map_filter="rank:2",     # omit for DEX-AR as published
)

visualize(image=image, heatmap=result.sentence_heatmap, title="Sentence heatmap")
```

Pass `target_sentence=None` to free-run instead: each token is the model's own
final-layer argmax, which is what the evaluation protocol uses.

`compute_dexar` returns a `DexarResult`:

| Attribute | Shape | Description |
|---|---|---|
| `per_token_heatmaps` | `[T, H, W]` | Per-token heatmaps, head filtering applied (Eq. 5) |
| `per_token_heatmaps_unfiltered` | `[T, H, W]` | Plain sum over heads and layers |
| `token_weights` | `[T]` | Visual relevance weight per token (Eq. 6) |
| `sentence_heatmap` | `[H, W]` | Sentence-level heatmap (Eq. 6) |
| `sentence_heatmap_unfiltered` | `[H, W]` | Unfiltered sentence-level heatmap |
| `tokens` | `list[str]` | Decoded token strings |

### Supported models

| Model | HuggingFace ID | env |
|---|---|---|
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` | either |
| BakLLaVA-v1 | `llava-hf/bakLlava-v1-hf` | either |
| Qwen3-VL-8B | `Qwen/Qwen3-VL-8B-Instruct` | `dexar` |
| LLaVA-OneVision-1.5-8B | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` | `dexar` |
| LLaVA-OneVision-2-8B | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | `dexar-tf5` |
| GLM-4.6V-Flash | `zai-org/GLM-4.6V-Flash` | `dexar-tf5` |

### Key parameters

- **`layer_index`** — starting depth for the gradient. `0` is all layers, which is
  the paper's Eq. 5; the original release defaulted to the last 10 (`-10`).
- **`map_filter`** — `"rank:k"`, the prefilter above. `None` = as published.
- **`drop_delta`** — set the Eq. 6 token weights to 1. Keep it off; dropping them
  costs SNR.
- **`prompt`** — for LLaVA, the full template carrying `<image>`; for the chat
  models, the bare instruction, wrapped in their own template.

## Acknowledgement

Built as a wrapper around [HuggingFace Transformers](https://github.com/huggingface/transformers),
on top of the original [DEX-AR](https://github.com/WalBouss/DEX-AR) release, which
takes inspiration from [LeGrad](https://github.com/WalBouss/LeGrad).

## Citation

If you use this code, please cite the original DEX-AR paper:

```bibtex
@article{bousselham2026dexar,
  author    = {Bousselham, Walid and Boggust, Angie and Strobelt, Hendrik and Kuehne, Hilde},
  title     = {DEX-AR: A Dynamic Explainability Method for Autoregressive Vision-Language Models},
  journal   = {arXiv preprint arXiv:2603.06302},
  year      = {2026},
}
```
