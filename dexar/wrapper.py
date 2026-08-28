from dataclasses import dataclass
from typing import Optional

import inspect

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from .utils import min_max, topk_norm


@dataclass
class DexarResult:
    """Output of DEX-AR explainability computation.

    Attributes:
        per_token_heatmaps: [num_tokens, H, W] filtered per-token heatmaps (Eq. 5 head filtering applied).
        per_token_heatmaps_unfiltered: [num_tokens, H, W] unfiltered per-token heatmaps (plain sum over all heads/layers).
        token_weights: [num_tokens] delta^t visual relevance weights (Eq. 6).
        sentence_heatmap: [H, W] filtered sentence-level heatmap (Eq. 6).
        sentence_heatmap_unfiltered: [H, W] unfiltered sentence-level heatmap.
        tokens: Decoded token strings.
        head_scores_img: [num_tokens, num_layers_used * heads] S_img, the per-head
            top-k norm over image positions (Eq. 4). Kept because delta^t and the
            head filter are both differences of S_img and S_text, so a zero
            delta^t is only interpretable if you can see which side won.
        head_scores_text: [num_tokens, num_layers_used * heads] S_text, the same
            statistic over the text positions selected by `text_set`.
    """
    per_token_heatmaps: torch.Tensor
    per_token_heatmaps_unfiltered: torch.Tensor
    token_weights: torch.Tensor
    sentence_heatmap: torch.Tensor
    sentence_heatmap_unfiltered: torch.Tensor
    tokens: list
    head_scores_img: Optional[torch.Tensor] = None
    head_scores_text: Optional[torch.Tensor] = None


class _LayerCapture:
    """Collects per-layer attention probabilities and hidden states via forward hooks.

    transformers >= 4.52 removed `output_attentions` / `output_hidden_states` from
    LLaVA and Qwen3-VL: the decoder layer computes its attention probabilities and
    then discards them (`hidden_states, _ = self.self_attn(...)`), returning a bare
    tensor. The attention module itself still *returns* `(attn_output, attn_weights)`,
    so a forward hook on it recovers the probabilities with the autograd graph
    intact — which is what DEX-AR differentiates through.

    Requires `attn_implementation="eager"`; the fused kernels never materialize
    the probability matrix at all.
    """

    def __init__(self, layers):
        self.attentions = {}
        self.hidden_states = {}
        self._handles = []
        for idx, layer in enumerate(layers):
            # Some eager implementations (LLaVA-OneVision-1.5) compute the
            # probabilities and then drop them -- `if not output_attentions:
            # attn_weights = None` -- so the forward hook captures None and the
            # only symptom is "got NoneType" from autograd.grad. Inject the flag
            # where the module's own signature accepts it; the newer decoders
            # (Qwen3-VL, GLM, OneVision-2) do not take the argument and return
            # the weights unconditionally, so they are left untouched.
            if "output_attentions" in inspect.signature(
                    layer.self_attn.forward).parameters:
                self._handles.append(
                    layer.self_attn.register_forward_pre_hook(
                        self._force_output_attentions, with_kwargs=True))
            self._handles.append(
                layer.self_attn.register_forward_hook(self._attention_hook(idx))
            )
            self._handles.append(
                layer.register_forward_hook(self._hidden_state_hook(idx))
            )

    @staticmethod
    def _force_output_attentions(module, args, kwargs):
        kwargs["output_attentions"] = True
        return args, kwargs

    def _attention_hook(self, idx):
        def hook(module, args, output):
            self.attentions[idx] = output[1]
        return hook

    def _hidden_state_hook(self, idx):
        def hook(module, args, output):
            self.hidden_states[idx] = output if torch.is_tensor(output) else output[0]
        return hook

    def clear(self):
        self.attentions.clear()
        self.hidden_states.clear()

    def remove(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


#: Definitions of the S_text token set used by Eq. 5 / Eq. 6.
#:
#: "all_non_image" is the published method. The others exist because on
#: chat-template models the scaffolding tokens dominate the text gradient:
#: on Qwen3-VL the 'assistant' role token wins for every generated token, and
#: removing it merely promotes 'user', then the instruction's final '.'.
TEXT_SET_MODES = (
    "all_non_image",         # every non-image position (published)
    "minus_special_suffix",  # drop special tokens and the generation-prompt tail
    "content_only",          # instruction words + generated tokens, no scaffolding
    "generated_only",        # previously generated tokens only
)


class _Architecture:
    """Model-family specific prompt building and image-token layout.

    Module paths (`model.model.language_model`, `model.lm_head`) are identical
    across LLaVA and Qwen3-VL on transformers >= 4.52, so only input construction
    and the visual token grid differ.
    """

    default_dtype = torch.float16

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def scaffolding_positions(self, input_ids) -> tuple:
        """Prompt positions as (special, generation-prompt suffix, semantic content).

        The base implementation knows only about tokenizer special tokens, which
        is all a raw-prompt model such as LLaVA has.
        """
        prompt_len = input_ids.shape[-1]
        special_ids = set(self.processor.tokenizer.all_special_ids)
        special = {i for i in range(prompt_len) if int(input_ids[0, i]) in special_ids}
        content = set(range(prompt_len)) - special
        return special, set(), content

    @property
    def image_token_id(self) -> int:
        config = self.model.config
        token_id = getattr(config, "image_token_id", None)
        if token_id is None:
            token_id = getattr(config, "image_token_index")
        return token_id

    def build_inputs(self, image: Image.Image, prompt: str) -> dict:
        raise NotImplementedError

    def vision_kwargs(self, inputs: dict) -> dict:
        """Extra kwargs the forward pass needs to encode the image."""
        return {"pixel_values": inputs["pixel_values"]}

    def step_kwargs(self, vision_kwargs: dict, seq_len: int) -> dict:
        """Adapt `vision_kwargs` to the current sequence length.

        DEX-AR re-runs a full forward pass per generation step with the sequence
        one token longer each time. Most vision kwargs are length-independent, so
        the default is a no-op; architectures carrying a per-position tensor
        (GLM-4V's `mm_token_type_ids`) must pad it here or the model will reject
        the shape mismatch on step 2.
        """
        return vision_kwargs

    def grid_size(self, inputs: dict, num_image_tokens: int) -> tuple:
        """Spatial (H, W) the image tokens unflatten to."""
        side = int(round(num_image_tokens ** 0.5))
        if side * side != num_image_tokens:
            raise ValueError(
                f"{num_image_tokens} image tokens do not form a square grid; "
                "this architecture needs an explicit grid_size implementation."
            )
        return side, side


class _LlavaArchitecture(_Architecture):
    """LLaVA-1.5 / BakLLaVA: fixed 24x24 grid, `<image>` placeholder in a raw prompt."""

    default_dtype = torch.float16
    default_prompt = "USER: <image>\nDescribe the image. ASSISTANT:"

    def build_inputs(self, image, prompt):
        if "<image>" not in prompt:
            raise ValueError("LLaVA prompts must contain the '<image>' placeholder.")
        return self.processor(text=prompt, images=image, return_tensors="pt")


class _Qwen3VLArchitecture(_Architecture):
    """Qwen3-VL: dynamic-resolution grid, chat-template prompt.

    The vision tower emits `(grid_h / merge) x (grid_w / merge)` tokens, where the
    grid comes from the processor's smart-resize, so both the token count and the
    heatmap resolution vary with the input image.
    """

    default_dtype = torch.bfloat16
    default_prompt = "Describe the image."

    def build_inputs(self, image, prompt):
        # Treat a prompt that already carries chat markup as fully formed.
        if "<|im_start|>" not in prompt:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.processor(text=[prompt], images=[image], return_tensors="pt")

    def vision_kwargs(self, inputs):
        return {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }

    def grid_size(self, inputs, num_image_tokens):
        merge = self.model.config.vision_config.spatial_merge_size
        _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
        return grid_h // merge, grid_w // merge

    def scaffolding_positions(self, input_ids):
        # Layout: <|im_start|>user \n <|vision_start|> IMAGE <|vision_end|>
        #         INSTRUCTION <|im_end|> \n <|im_start|>assistant \n
        special, _, _ = super().scaffolding_positions(input_ids)
        tokenizer = self.processor.tokenizer
        prompt_len = input_ids.shape[-1]

        starts = (input_ids[0] == tokenizer.convert_tokens_to_ids("<|im_start|>")).nonzero()
        suffix = set(range(int(starts[-1]), prompt_len)) if starts.numel() else set()

        vision_end = (input_ids[0] == tokenizer.convert_tokens_to_ids("<|vision_end|>")).nonzero()
        im_end = (input_ids[0] == tokenizer.convert_tokens_to_ids("<|im_end|>")).nonzero()
        if vision_end.numel() and im_end.numel():
            content = set(range(int(vision_end[-1]) + 1, int(im_end[0])))
        else:
            content = set(range(prompt_len)) - special - suffix
        return special, suffix, content


class _OneVisionArchitecture(_Qwen3VLArchitecture):
    """LLaVA-OneVision 1.5 / 2: Qwen ChatML prompting, non-Qwen vision tower.

    Both checkpoints reuse Qwen's tokenizer wholesale -- `<|vision_start|>` 151652,
    `<|image_pad|>` 151655, `<|im_start|>` 151644 -- and both feed a
    `Qwen2VLImageProcessor`, so the grid, the vision kwargs and the scaffolding
    split are all Qwen3-VL's. Only the prompt differs: neither ships a chat
    template (no `chat_template.json`, nothing in `tokenizer_config.json`), so
    `apply_chat_template` raises and the ChatML has to be written out here.

    That is the point of running them. OneVision-2's text_config is `qwen3` at
    36 layers and 32/8 GQA -- Qwen3-VL's language model, behind a different
    vision encoder. If the value-norm register is a property of the LM it should
    survive; if it is induced by the vision tower it should not.
    """

    default_dtype = torch.bfloat16
    default_prompt = "Describe the image."

    def vision_kwargs(self, inputs):
        kw = super().vision_kwargs(inputs)
        # OneVision-2's encoder takes explicit per-patch positions and has no
        # fallback: `forward_from_positions` dereferences them unconditionally,
        # so dropping the key crashes rather than degrading. OneVision-1.5's
        # processor does not emit them, hence the presence check.
        if "patch_positions" in inputs:
            kw["patch_positions"] = inputs["patch_positions"]
        return kw

    def build_inputs(self, image, prompt):
        if "<|im_start|>" not in prompt:
            prompt = ("<|im_start|>user\n"
                      "<|vision_start|><|image_pad|><|vision_end|>"
                      f"{prompt}<|im_end|>\n"
                      "<|im_start|>assistant\n")
        return self.processor(text=[prompt], images=[image], return_tensors="pt")


class _Glm4vArchitecture(_Architecture):
    """GLM-4.1V / 4.5V / 4.6V: dynamic-resolution grid, GLM chat template.

    Structurally a Qwen3-VL clone from DEX-AR's point of view -- same module
    paths, same `image_grid_thw` contract, same merge-size arithmetic -- so the
    only real differences are the prompt markup and the scaffolding layout:

        [gMASK]<sop><|user|>\n<|begin_of_image|> IMAGE <|end_of_image|>
        INSTRUCTION <|assistant|>\n

    NOTE the 4.6V checkpoint declares `Glm46VProcessor` / `Glm46VImageProcessor`,
    which do not exist in transformers 4.57.6, and ships no remote code for them.
    `AutoProcessor` therefore falls back to the bare tokenizer *silently* -- the
    object it returns has no `image_processor` and raises only when you pass
    images. `from_pretrained` below assembles a `Glm4vProcessor` by hand instead;
    the 4.1V/4.5V processor is config-compatible with 4.6V.
    """

    default_dtype = torch.bfloat16
    default_prompt = "Describe the image."

    IMG_START, IMG_END = 151339, 151340

    def __init__(self, model, processor):
        super().__init__(model, processor)
        self._install_mm_token_type_shim()

    def _install_mm_token_type_shim(self):
        """Keep `mm_token_type_ids` in step with `input_ids` on EVERY forward.

        transformers >= 5.0 requires this per-position tensor for M-RoPE, and the
        processor builds it once at prompt length. Any caller that re-runs the
        model with a grown sequence dies with a shape mismatch on step 2 -- that
        is compute_dexar AND ~20 direct `w.model.model(...)` sites across
        experiments/. Patching the inner model's forward fixes all of them at
        once, including scripts written later, instead of each call site.
        """
        inner = self.model.model
        if getattr(inner, "_dexar_mm_shim", False):
            return
        orig = inner.forward

        def shim(*args, **kwargs):
            ids = kwargs.get("input_ids")
            mm = kwargs.get("mm_token_type_ids")
            if ids is not None and mm is not None and mm.shape[-1] != ids.shape[-1]:
                delta = ids.shape[-1] - mm.shape[-1]
                if delta > 0:      # generated tokens are text -> type 0
                    pad = torch.zeros((mm.shape[0], delta),
                                      dtype=mm.dtype, device=mm.device)
                    kwargs["mm_token_type_ids"] = torch.cat([mm, pad], dim=-1)
                else:
                    kwargs["mm_token_type_ids"] = mm[:, :ids.shape[-1]]
            return orig(*args, **kwargs)

        inner.forward = shim
        inner._dexar_mm_shim = True

    def build_inputs(self, image, prompt):
        if "<|user|>" not in prompt:
            messages = [{
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": prompt}],
            }]
            prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return self.processor(text=[prompt], images=[image], return_tensors="pt")

    def vision_kwargs(self, inputs):
        kw = {
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }
        # transformers >= 5.0 requires this for M-RoPE; 4.57.x neither needs nor
        # emits it, so pass it only when the processor produced one.
        if "mm_token_type_ids" in inputs:
            kw["mm_token_type_ids"] = inputs["mm_token_type_ids"]
        return kw

    def step_kwargs(self, vision_kwargs, seq_len):
        mm = vision_kwargs.get("mm_token_type_ids")
        if mm is None or mm.shape[-1] == seq_len:
            return vision_kwargs
        # Generated tokens are text, so type 0. Pad on the right to match.
        pad = torch.zeros((mm.shape[0], seq_len - mm.shape[-1]),
                          dtype=mm.dtype, device=mm.device)
        return {**vision_kwargs, "mm_token_type_ids": torch.cat([mm, pad], dim=-1)}

    def grid_size(self, inputs, num_image_tokens):
        merge = self.processor.image_processor.merge_size
        _, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
        return grid_h // merge, grid_w // merge

    def scaffolding_positions(self, input_ids):
        special, _, _ = super().scaffolding_positions(input_ids)
        tokenizer = self.processor.tokenizer
        prompt_len = input_ids.shape[-1]

        asst = tokenizer.convert_tokens_to_ids("<|assistant|>")
        hits = (input_ids[0] == asst).nonzero()
        suffix = set(range(int(hits[-1]), prompt_len)) if hits.numel() else set()

        img_end = (input_ids[0] == self.IMG_END).nonzero()
        if img_end.numel() and hits.numel():
            content = set(range(int(img_end[-1]) + 1, int(hits[-1])))
        else:
            content = set(range(prompt_len)) - special - suffix
        return special, suffix, content


def _select_architecture(model, processor) -> _Architecture:
    """Dispatch on model_type first, class name second.

    Prefix matching on the class name alone is unsafe: LLaVA-OneVision-2 is
    `LlavaOnevision2ForConditionalGeneration`, which `startswith("Llava")`
    matches -- so it would silently run under the LLaVA-1.5 adapter, with raw
    `<image>` prompting and a square-grid assumption, on a dynamic-resolution
    model. Unsupported architectures must fail loudly, not quietly.
    """
    name = type(model).__name__
    model_type = getattr(model.config, "model_type", "") or ""
    if model_type.startswith("qwen3_vl") or name.startswith("Qwen3VL"):
        return _Qwen3VLArchitecture(model, processor)
    if model_type.startswith("llava_onevision2") or name in (
            "LlavaOnevision2ForConditionalGeneration",
            "LLaVAOneVision1_5_ForConditionalGeneration"):
        return _OneVisionArchitecture(model, processor)
    if model_type.startswith("glm4v") or name.startswith("Glm4v"):
        return _Glm4vArchitecture(model, processor)
    if model_type == "llava" or name == "LlavaForConditionalGeneration":
        return _LlavaArchitecture(model, processor)
    raise ValueError(
        f"Unsupported architecture: {name} (model_type={model_type!r}). "
        "Add an _Architecture subclass -- do NOT rely on class-name prefixes."
    )




class DexarWrapper:
    """Wraps a HuggingFace vision-language model for DEX-AR explainability.

    Supports LLaVA-1.5 / BakLLaVA and Qwen3-VL.

    Args:
        model: A LlavaForConditionalGeneration or Qwen3VLForConditionalGeneration model.
        processor: The corresponding AutoProcessor.
        layer_index: Which layers to read. An int is the starting depth
            (negative = from end); a (start, end) tuple is a half-open band; a
            sequence is exactly those layers. Default -10.
        text_set: Which positions count as text when scoring S_text. Defaults to
            "all_non_image", the published method. See TEXT_SET_MODES.
    """

    def __init__(self, model, processor, layer_index=-10,
                 text_set: str = "all_non_image"):
        if text_set not in TEXT_SET_MODES:
            raise ValueError(f"text_set must be one of {TEXT_SET_MODES}, got {text_set!r}")
        self.model = model
        self.processor = processor
        self.layer_index = layer_index
        self.text_set = text_set
        self.arch = _select_architecture(model, processor)
        self._assert_eager_attention()

        if model.config._attn_implementation != "eager":
            raise ValueError(
                "DEX-AR needs the attention probability matrix, which only the "
                f"eager implementation materializes (got "
                f"'{model.config._attn_implementation}'). Load the model with "
                "attn_implementation='eager'."
            )

        # Freeze all params, enable gradients only on language_model
        for name, param in self.model.named_parameters():
            param.requires_grad = "language_model" in name

        # Store model internals
        self.language_model = model.model.language_model
        self.num_layers = len(self.language_model.layers)
        self.lm_head = model.lm_head
        self.norm = self.language_model.norm
        self.layers = self._resolve_layers(layer_index)

    def _resolve_layers(self, layer_index) -> list:
        """Layer indices whose gradients are accumulated.

        int            -> [layer_index, num_layers), negative counts from the end.
                          This is the published behaviour ("starting depth").
        (start, end)    -> explicit half-open band, e.g. (22, 27).
        sequence of int -> exactly those layers, e.g. [24].

        Which layers are read matters more than any other setting on Qwen3-VL:
        reading at layer 24 gives markedly more concept-discriminative maps than
        the deeper layers where the logit lens is most rank-accurate.
        """
        def normalize(i):
            i = i if i >= 0 else self.num_layers + i
            if not 0 <= i < self.num_layers:
                raise ValueError(
                    f"layer {i} out of range for a {self.num_layers}-layer model")
            return i

        if isinstance(layer_index, int):
            return list(range(normalize(layer_index), self.num_layers))
        if isinstance(layer_index, tuple) and len(layer_index) == 2:
            start, end = normalize(layer_index[0]), normalize(layer_index[1] - 1) + 1
            if start >= end:
                raise ValueError(f"empty layer band: {layer_index}")
            return list(range(start, end))
        if isinstance(layer_index, (list, range)):
            layers = sorted({normalize(i) for i in layer_index})
            if not layers:
                raise ValueError("layer_index sequence is empty")
            return layers
        raise TypeError(
            "layer_index must be an int, a (start, end) tuple, or a sequence of ints; "
            f"got {type(layer_index).__name__}"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        layer_index=-10,
        dtype: Optional[torch.dtype] = None,
        text_set: str = "all_non_image",
    ) -> "DexarWrapper":
        """Load a vision-language model and processor from HuggingFace.

        Supports LLaVA-1.5, BakLLaVA and Qwen3-VL model variants.

        Args:
            model_name: HuggingFace model ID (e.g. "llava-hf/llava-1.5-7b-hf",
                "Qwen/Qwen3-VL-8B-Instruct").
            device: Device to load model on. Default "cuda".
            layer_index: Starting depth (int), a (start, end) band, or an explicit
                sequence of layers. Default -10. On Qwen3-VL, reading a narrow band
                around layer 24 is markedly more concept-discriminative than the
                default; see docs/qwen3vl.md.
            dtype: Override the per-architecture default (fp16 for LLaVA,
                bf16 for Qwen3-VL).
            text_set: Which positions count as text when scoring S_text.
                Defaults to the published "all_non_image".
        """
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "") or ""
        is_qwen3vl = model_type.startswith("qwen3_vl")
        is_glm4v = model_type.startswith("glm4v")
        # The OneVision checkpoints ship their own modeling code and declare no
        # top-level model_type, so identify them by the architectures entry.
        archs = getattr(config, "architectures", None) or []
        is_onevision = model_type.startswith("llava_onevision2") or any(
            a in ("LlavaOnevision2ForConditionalGeneration",
                  "LLaVAOneVision1_5_ForConditionalGeneration") for a in archs)
        if dtype is None:
            dtype = (torch.bfloat16 if (is_qwen3vl or is_glm4v or is_onevision)
                     else torch.float16)

        if "LLaVAOneVision1_5_ForConditionalGeneration" in archs:
            # Its bundled code reads `config.text_config.pad_token_id`, which
            # transformers 4.x supplied implicitly on every PretrainedConfig and
            # 5.x no longer does. Supply the attribute rather than the value: the
            # checkpoint has no pad token, and inventing an id would silently
            # mask real positions. None reproduces the 4.x default exactly.
            for attr, default in (("pad_token_id", None),):
                if not hasattr(config.text_config, attr):
                    setattr(config.text_config, attr, default)

        if is_onevision:
            # Passing `config=` explicitly stops transformers from pushing
            # attn_implementation down into the sub-configs, and these
            # checkpoints build their decoder as
            # ATTENTION_CLASSES[text_config._attn_implementation]. Left alone the
            # LM silently runs SDPA, returns attentions=None, and DEX-AR dies in
            # the backward pass with "got NoneType" -- never at load time.
            config._attn_implementation = "eager"
            if hasattr(config, "text_config"):
                config.text_config._attn_implementation = "eager"
                # `_from_config` re-derives from the private slot, so setting the
                # public name alone is silently discarded on some checkpoints.
                config.text_config._attn_implementation_internal = "eager"

        loader = AutoModelForImageTextToText
        if "LLaVAOneVision1_5_ForConditionalGeneration" in archs:
            # LLaVA-OneVision-1.5 registers no model_type and no auto_map, so its
            # remote config maps to no AutoModel class. Its own inference.py uses
            # AutoModelForCausalLM; anything else raises "Unrecognized
            # configuration class" before a single weight is read.
            from transformers import AutoModelForCausalLM
            loader = AutoModelForCausalLM
        model = loader.from_pretrained(
            model_name,
            config=config if is_onevision else None,
            attn_implementation="eager",
            dtype=dtype,
            device_map=device,
            trust_remote_code=is_onevision,
        )
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=is_onevision)
        if is_glm4v and not hasattr(processor, "image_processor"):
            # GLM-4.6V's config names `Glm46VProcessor`, which exists only from
            # transformers 5.x, and the checkpoint ships no remote code for it.
            # On 4.57.x AutoProcessor therefore degrades to the bare TOKENIZER --
            # silently, raising only once images are passed. Rebuild from the
            # 4.1V/4.5V classes, which are config-compatible.
            #
            # This fallback is a stopgap, NOT equivalent: transformers 4.57.x also
            # has no rope_scaling for this checkpoint, and 5.x supplies
            # rope_theta=500000 where GLM-4.1V uses 10000. Running 4.6V on 4.57.x
            # would need that fabricated, so the model is gated below instead.
            from transformers import (AutoTokenizer, Glm4vImageProcessor,
                                      Glm4vProcessor, Glm4vVideoProcessor)
            _tok = AutoTokenizer.from_pretrained(model_name)
            processor = Glm4vProcessor(
                image_processor=Glm4vImageProcessor.from_pretrained(model_name),
                tokenizer=_tok,
                video_processor=Glm4vVideoProcessor.from_pretrained(model_name),
                chat_template=_tok.chat_template,
            )
        if is_glm4v:
            _rope = getattr(getattr(config, "text_config", config), "rope_scaling", None)
            if not _rope or "mrope_section" not in _rope:
                import transformers as _tf
                raise RuntimeError(
                    f"{model_name} has no rope_scaling under transformers "
                    f"{_tf.__version__}. Its config targets transformers 5.x, which "
                    "supplies mrope_section/partial_rotary_factor/rope_theta. Do NOT "
                    "backfill these from GLM-4.1V: 5.x gives rope_theta=500000 where "
                    "4.1V uses 10000, so guessing silently corrupts every positional "
                    "result. Run this model from the side env: ~/.venvs/dexar-tf5."
                )

        if not (is_qwen3vl or is_glm4v or is_onevision):
            # The revisions this repo used to pin predate `num_additional_image_tokens`,
            # so they expand `<image>` one token short of what the vision tower emits,
            # which transformers >= 4.52 rejects outright.
            processor.patch_size = model.config.vision_config.patch_size
            processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy

        return cls(model, processor, layer_index=layer_index, text_set=text_set)

    def _assert_eager_attention(self):
        """DEX-AR differentiates attention probabilities, so they must exist.

        Under SDPA the decoder returns `attentions=None` and the only symptom is
        `RuntimeError: all inputs have to be Tensors or GradientEdges, but got
        NoneType` from `torch.autograd.grad` -- raised after a full forward pass,
        with nothing naming attention. Checking the built module here turns that
        into a load-time error that says what is wrong.
        """
        layers = getattr(getattr(self.model, "model", None), "language_model", None)
        layers = getattr(layers, "layers", None)
        if not layers:
            return                                   # unknown layout; nothing to check
        attn = type(getattr(layers[0], "self_attn", None)).__name__
        if "Sdpa" in attn or "Flash" in attn:
            raise RuntimeError(
                f"language model built {attn}: DEX-AR needs eager attention, and "
                f"this returns attentions=None (the failure would surface only in "
                f"the backward pass). The checkpoint ignored "
                f"attn_implementation='eager' -- set it on config.text_config "
                f"before loading, including _attn_implementation_internal.")

    def _text_positions(self, seq_len, prompt_len, image_positions, scaffolding, device):
        """Positions scored as text by Eq. 5 / Eq. 6, per `self.text_set`.

        Generated tokens always count as text; only the prompt side varies.
        """
        special, suffix, content = scaffolding
        generated = range(prompt_len, seq_len)

        if self.text_set == "generated_only":
            positions = list(generated)
        elif self.text_set == "content_only":
            positions = sorted(content) + list(generated)
        else:
            excluded = set(image_positions.tolist())
            if self.text_set == "minus_special_suffix":
                excluded |= special | suffix
            positions = [i for i in range(seq_len) if i not in excluded]

        if not positions:
            # e.g. generated_only on the first step: no text to compare against
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.tensor(positions, dtype=torch.long, device=device)

    @staticmethod
    def _rank_filter(x, rank, h, w, size=3):
        """k-th order statistic of each cell's size x size neighbourhood.

        x: [heads, N] over image positions, N = h*w. rank 0 is erosion, the
        middle index is the median, size^2-1 is dilation. Edges replicate --
        zero padding would erode the border as a padding artifact, which on a
        model with [[border-peak-bias]] would look like the method working.

        Eq. 5 scores a head by topk_norm(., k=1), i.e. by its single tallest
        cell. A sub-median rank filter is exactly the operator an isolated cell
        cannot survive (its neighbourhood holds size^2-1 small values) but a
        coherent region can, so this rewrites S_img from "the tallest cell" into
        "the tallest cell that has neighbours" without touching Eq. 5.
        """
        g = x.reshape(x.shape[0], 1, h, w)
        p = F.pad(g, (size // 2,) * 4, mode="replicate")
        u = F.unfold(p, kernel_size=size)                 # [heads, size^2, N]
        return u.sort(dim=1).values[:, rank, :]           # [heads, N]


    @torch.inference_mode(False)
    def compute_dexar(
        self,
        image: Image.Image,
        target_sentence: Optional[str] = None,
        prompt: Optional[str] = None,
        max_new_tokens: int = 20,
        map_filter: Optional[str] = None,
        drop_delta: bool = False,
    ) -> DexarResult:
        """Compute DEX-AR explainability maps.

        Args:
            image: Input PIL Image.
            target_sentence: The sentence to explain, teacher-forced. Pass None to
                free-run instead: at each step the token is the model's own
                final-layer argmax, which is the "sampled word" of Eq. 2/3. The
                per-layer logit lens then reads that same vocabulary id.
            prompt: Prompt for the model. For LLaVA this is the full template
                containing `<image>`; for Qwen3-VL it is the user instruction,
                wrapped in the chat template unless it already carries one.
                Defaults to the architecture's own template.
            max_new_tokens: Step cap when free-running. Ignored when a
                target_sentence is given.
            map_filter: Spatial prefilter applied to each (token, layer, head)
                image map BEFORE Eq. 5 scores it. Spec 'rank:k' -- the k-th
                order statistic of a 3x3 neighbourhood (k=4 is a median, k=0
                erosion, k=8 dilation). S_img is rescaled per (token, layer) by
                median_heads(S_img_orig)/median_heads(S_img_filt), because
                S_text lives on the 1-D token sequence and receives no matching
                operation -- without that the subtraction in Eq. 5 compares a
                filtered quantity against an unfiltered one and the head weights
                collapse. None = off (published method).
            drop_delta: Set the Eq. 6 token weights delta^t to 1 instead of
                (max S_img - max S_text)+. delta^t is a second max-vs-max
                comparison, and any prefilter that lowers the max kills it: on
                cat_and_dog.jpg it is already 1/15 unfiltered and 0/15 after.
                Off by default (published method).

        Returns:
            DexarResult with per-token heatmaps, token weights, and sentence heatmap.
        """
        device = next(self.model.parameters()).device
        if prompt is None:
            prompt = self.arch.default_prompt

        # --- Tokenize prompt and target ---
        inputs = self.arch.build_inputs(image, prompt).to(device)
        input_ids = inputs["input_ids"]
        vision_kwargs = self.arch.vision_kwargs(inputs)

        tokenizer = self.processor.tokenizer
        free_run = target_sentence is None
        if free_run:
            # Tokens are not known ahead of the loop; each is chosen at its step.
            target_ids = None
            max_steps = max_new_tokens
            token_strings = []
            eos_ids = {i for i in (tokenizer.eos_token_id,
                                   getattr(tokenizer, "pad_token_id", None))
                       if i is not None}
        else:
            target_ids = tokenizer.encode(
                target_sentence, add_special_tokens=False, return_tensors="pt"
            ).to(device)
            max_steps = target_ids.shape[-1]
            # Decode individual tokens for output
            token_strings = [tokenizer.decode(target_ids[0, i]) for i in range(max_steps)]
            eos_ids = set()

        # --- Locate the image tokens ---
        # Read them off the sequence rather than by counting prompt characters:
        # Qwen3-VL's token count varies per image, and the arithmetic this replaces
        # dropped one text token off the end of the image block.
        image_positions = (input_ids[0] == self.arch.image_token_id).nonzero().flatten()
        if image_positions.numel() == 0:
            raise ValueError("No image tokens found in the tokenized prompt.")
        num_img_tokens = image_positions.numel()
        h, w = self.arch.grid_size(inputs, num_img_tokens)
        prompt_len = input_ids.shape[-1]
        scaffolding = self.arch.scaffolding_positions(input_ids)

        # --- Main token generation loop ---
        attention_mask = torch.ones_like(input_ids, device=device)
        all_new_tokens_grads = []
        all_heads_topk_norm_img = []
        all_heads_topk_norm_text = []

        capture = _LayerCapture(self.language_model.layers)
        try:
            for n in range(max_steps):
                capture.clear()
                # Call the inner model: it produces the hidden states the hooks read,
                # without running lm_head over every position.
                self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    **self.arch.step_kwargs(vision_kwargs, input_ids.shape[-1]),
                )

                attentions = capture.attentions
                hidden_states = capture.hidden_states

                if free_run:
                    # Eq. 2: the sampled word is the final layer's argmax.
                    with torch.no_grad():
                        final_logits = self.lm_head(
                            self.norm(hidden_states[self.num_layers - 1][:, -1]))
                    next_token_id = final_logits.argmax(dim=-1)
                    if int(next_token_id) in eos_ids:
                        break
                    token_strings.append(tokenizer.decode(next_token_id[0]))
                else:
                    next_token_id = target_ids[:, n]

                seq_len = input_ids.shape[-1]
                text_positions = self._text_positions(
                    seq_len, prompt_len, image_positions, scaffolding, device
                )

                new_token_grads = []
                heads_topk_norm_img = []
                heads_topk_norm_text = []

                # Head scoring uses max mode (k=1) — Table 3 best
                k_img = 1
                k_all_text = 1

                for l in self.layers:
                    # Intermediate logits via logit lens
                    interm_logits = self.lm_head(self.norm(hidden_states[l][:, -1]))

                    one_hot = interm_logits[:, next_token_id].sum()
                    grad = torch.autograd.grad(
                        one_hot, [attentions[l]], retain_graph=True
                    )[0]

                    # Gradient of the newly predicted token w.r.t. every attended position
                    grad_last = grad[:, :, -1, :]
                    grad_img = grad_last[:, :, image_positions].clamp(min=0.0)

                    simg_scale = 1.0
                    if map_filter is not None:
                        bits = map_filter.split(":")
                        if bits[0] != "rank" or len(bits) not in (2, 3):
                            raise ValueError(
                                f"unknown map_filter {map_filter!r}; expected "
                                f"'rank:k' or 'rank:k:noscale'")
                        gi = grad_img.detach()
                        filt = self._rank_filter(gi[0], int(bits[1]), h, w)[None]
                        if len(bits) == 3 and bits[2] == "noscale":
                            # ablation: skip the S_img rescale, so Eq. 5 compares a
                            # filtered S_img against an unfiltered S_text.
                            simg_scale = 1.0
                        else:
                            s_o = topk_norm(gi, k=1).median(dim=-1, keepdim=True).values
                            s_f = topk_norm(filt, k=1).median(dim=-1, keepdim=True).values
                            simg_scale = s_o / s_f.clamp(min=1e-12)
                        grad_img = filt

                    new_token_grads.append(grad_img.detach())
                    heads_topk_norm_img.append(topk_norm(grad_img, k=k_img) * simg_scale)
                    if text_positions.numel():
                        grad_all_text = grad_last[:, :, text_positions].clamp(min=0)
                        heads_topk_norm_text.append(topk_norm(grad_all_text, k=k_all_text))
                    else:
                        # No text to compare against, so nothing suppresses the image score.
                        heads_topk_norm_text.append(torch.zeros_like(heads_topk_norm_img[-1]))

                # Aggregate across layers for this token
                all_heads_topk_norm_img.append(torch.cat(heads_topk_norm_img))
                all_heads_topk_norm_text.append(torch.cat(heads_topk_norm_text))
                new_token_grads = torch.cat(new_token_grads)  # [num_layers_used, heads, N]
                all_new_tokens_grads.append(new_token_grads)

                # Extend input sequence with the current token
                input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), device=device, dtype=attention_mask.dtype)],
                    dim=1,
                )
        finally:
            capture.remove()

        # ========== Aggregation ==========
        # Free-running can stop early on EOS, so the step count is only known now.
        num_tokens_to_generate = len(all_new_tokens_grads)
        if num_tokens_to_generate == 0:
            raise ValueError("no tokens were generated (EOS on the first step)")

        # [num_tokens, num_layers_used, heads, N]
        all_new_tokens_grads = torch.stack(all_new_tokens_grads)
        all_new_tokens_grads = min_max(all_new_tokens_grads) # FIX
        # [num_tokens, num_layers_used * heads]
        all_heads_topk_norm_img = torch.stack(all_heads_topk_norm_img)
        all_heads_topk_norm_text = torch.stack(all_heads_topk_norm_text)

        # --- Head filtering (Eq. 5): w = (S_img - S_text)^+ ---
        filtering_weights = (all_heads_topk_norm_img - all_heads_topk_norm_text).clamp(min=0)
        # [num_tokens, num_layers_used, heads]
        num_layers_used = len(self.layers)
        num_heads = all_new_tokens_grads.shape[2]
        filtering_weights = filtering_weights.view(
            num_tokens_to_generate, num_layers_used, num_heads
        )

        # --- Filtered per-token heatmaps (Eq. 5): weighted by head filtering ---
        filtered_grads = all_new_tokens_grads * filtering_weights.unsqueeze(-1)
        per_token_flat = filtered_grads.sum(dim=[1, 2])  # [num_tokens, N]
        per_token_heatmaps = rearrange(per_token_flat, "t (h w) -> t h w", h=h, w=w)
        for t in range(num_tokens_to_generate):
            hm = per_token_heatmaps[t]
            if hm.max() > hm.min():
                per_token_heatmaps[t] = min_max(hm)

        # --- Unfiltered per-token heatmaps: plain sum over all heads/layers ---
        unfiltered_flat = all_new_tokens_grads.sum(dim=[1, 2])  # [num_tokens, N]
        per_token_heatmaps_unfiltered = rearrange(unfiltered_flat, "t (h w) -> t h w", h=h, w=w)
        for t in range(num_tokens_to_generate):
            hm = per_token_heatmaps_unfiltered[t]
            if hm.max() > hm.min():
                per_token_heatmaps_unfiltered[t] = min_max(hm)

        # --- Token weights delta^t (Eq. 6) ---
        num_gen_tokens = all_heads_topk_norm_img.shape[0]
        delta_t = (
            all_heads_topk_norm_img.view(num_gen_tokens, -1).max(dim=-1)[0]
            - all_heads_topk_norm_text.view(num_gen_tokens, -1).max(dim=-1)[0]
        ).clamp(min=0)  # [num_tokens]
        if drop_delta:
            delta_t = torch.ones_like(delta_t)

        # --- Filtered sentence-level heatmap (Eq. 6) ---
        sentence_flat = (
            filtered_grads * delta_t[:, None, None, None]
        ).sum(dim=[0, 1, 2])  # [N]
        sentence_heatmap = rearrange(sentence_flat, "(h w) -> h w", h=h, w=w)
        if sentence_heatmap.max() > sentence_heatmap.min():
            sentence_heatmap = min_max(sentence_heatmap)

        # --- Unfiltered sentence-level heatmap ---
        unfiltered_sentence_flat = all_new_tokens_grads.sum(dim=[0, 1, 2])  # [N]
        sentence_heatmap_unfiltered = rearrange(unfiltered_sentence_flat, "(h w) -> h w", h=h, w=w)
        if sentence_heatmap_unfiltered.max() > sentence_heatmap_unfiltered.min():
            sentence_heatmap_unfiltered = min_max(sentence_heatmap_unfiltered)

        return DexarResult(
            per_token_heatmaps=per_token_heatmaps.detach(),
            per_token_heatmaps_unfiltered=per_token_heatmaps_unfiltered.detach(),
            token_weights=delta_t.detach(),
            sentence_heatmap=sentence_heatmap.detach(),
            sentence_heatmap_unfiltered=sentence_heatmap_unfiltered.detach(),
            head_scores_img=all_heads_topk_norm_img.detach(),
            head_scores_text=all_heads_topk_norm_text.detach(),
            tokens=token_strings,
        )
