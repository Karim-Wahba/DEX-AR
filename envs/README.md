# Environments

Two conda envs. Neither transformers version runs all five models.

| file | transformers | models |
|---|---|---|
| `dexar.yml` | 4.57.6 | LLaVA-1.5, BakLLaVA, Qwen3-VL, LLaVA-OneVision-1.5 |
| `dexar-tf5.yml` | 5.15.1 | GLM-4.6V-Flash, LLaVA-OneVision-2 |

Both are Python 3.10.20 with torch 2.13.0+cu130, exported with
`conda env export --no-builds` (the `prefix:` line is stripped, so they carry no
absolute path).

```bash
conda env create -f envs/dexar.yml
conda env create -f envs/dexar-tf5.yml
conda activate dexar && pip install -e .     # and again in dexar-tf5
```

`pip install -e .` is not in the yml files: it would encode this checkout's path.

## Why each model is pinned where it is

- **GLM-4.6V-Flash** — its config targets transformers 5 and omits `rope_scaling`,
  so it cannot load on 4.57.
- **LLaVA-OneVision-2** — needs transformers 5; its encoder dereferences
  `patch_positions`, which the 4.x path does not pass through.
- **LLaVA-OneVision-1.5** — ships transformers-4 modeling code. Transformers 5
  dropped the `'default'` key from `ROPE_INIT_FUNCTIONS`, which its
  `RotaryEmbedding` indexes unconditionally, so it must stay on 4.57.
- **Qwen3-VL, LLaVA-1.5, BakLLaVA** — run on either. The two envs were checked
  against each other on BakLLaVA and give identical VOC metrics.

`attn_implementation="eager"` is required and the wrapper sets it: the fused
attention kernels never materialise the probability matrix DEX-AR
differentiates through.
