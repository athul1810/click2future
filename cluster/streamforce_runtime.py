"""Memory-bounded loading of the published StreamForce inference pipeline.

Only construction/loading is replaced. Forward methods, attention, force
conditioning, the released EMA tensors and the upstream sampler are retained.
The optional captioner is not loaded. Run in the external pinned environment.
"""

import gc
import json
import os
import sys
from pathlib import Path


def clean_state(state):
    prefixes = ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module.", "_orig_mod.")
    result = {}
    for name, value in state.items():
        for prefix in prefixes:
            name = name.replace(prefix, "")
        if name in result:
            raise ValueError("Checkpoint wrapper names collide")
        result[name] = value
    return result


def assign_state(module, state, *, device, dtype):
    """Strictly assign one tensor at a time without a second CPU model copy."""
    import torch

    expected = module.state_dict()
    if expected.keys() != state.keys():
        missing = sorted(expected.keys() - state.keys())
        extra = sorted(state.keys() - expected.keys())
        raise ValueError(f"Incomplete released state: missing={missing[:4]}, extra={extra[:4]}")
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or expected[name].shape != value.shape:
            raise ValueError(f"Invalid checkpoint tensor: {name}")
    loaded = {
        name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
        for name, value in state.items()
    }
    module.load_state_dict(loaded, strict=True, assign=True)
    if any(tensor.is_meta for tensor in module.state_dict().values()):
        raise RuntimeError("A checkpoint left unresolved meta tensors")
    return len(loaded)


def load_backend(repo, prompts, *, progress=lambda *_: None):
    import torch
    from omegaconf import OmegaConf

    repo = Path(repo).resolve()
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "demo")]
    from interactive_demo_backend_rolling_forcing import DemoBackendConfig, InteractiveDemoBackend
    from pipeline.rolling_forcing_streaming_inference import (
        ControlRollingForcingStreamingInferencePipeline,
    )
    from utils.wan_wrapper import (
        CausalWanControlNetWrapper,
        WanTextEncoder,
        WanVAEWrapper,
    )
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    base = repo / "wan_models/Wan2.2-TI2V-5B"
    progress("loading_text_encoder")
    encoder = WanTextEncoder.__new__(WanTextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.text_encoder = umt5_xxl(
        encoder_only=True, return_tokenizer=False, dtype=torch.bfloat16, device="meta"
    )
    text_state = torch.load(
        base / "models_t5_umt5-xxl-enc-bf16.pth",
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    text_tensors = assign_state(
        encoder.text_encoder, text_state, device=device, dtype=torch.bfloat16
    )
    del text_state
    encoder.tokenizer = HuggingfaceTokenizer(
        name=str(base / "google/umt5-xxl"), seq_len=512, clean="whitespace"
    )
    encoder.eval().requires_grad_(False)
    encoded = {}
    for prompt in dict.fromkeys(prompts):
        encoded[prompt] = encoder([prompt])["prompt_embeds"].detach().cpu()
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    progress("text_embeddings_ready")

    class CachedTextEncoder(torch.nn.Module):
        def __init__(self, embeddings):
            super().__init__()
            self.names = {}
            for index, (prompt, embedding) in enumerate(embeddings.items()):
                name = f"prompt_{index}"
                self.names[prompt] = name
                self.register_buffer(name, embedding)

        def forward(self, text_prompts):
            return {
                "prompt_embeds": torch.cat(
                    [getattr(self, self.names[prompt]) for prompt in text_prompts], dim=0
                )
            }

    args = OmegaConf.merge(
        OmegaConf.load(repo / "configs/default_config.yaml"),
        OmegaConf.load(repo / "configs/dmd_everything.yaml"),
    )
    args.gradient_checkpointing = False
    args.generator_model_kwargs.rolling_forcing_attention = True
    args.generator_model_kwargs.rolling_forcing_block_frames = 3
    args.generator_model_kwargs.rolling_forcing_max_frames = 21
    args.rolling_forcing_cache_frames = 96
    progress("loading_generator")
    generator = CausalWanControlNetWrapper(
        **args.generator_model_kwargs, is_causal=True, training=False, meta_init=True
    )
    checkpoint = repo / "checkpoints/streamforce_v1_ckpt.pt"
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    if "generator_ema" not in state:
        raise ValueError("The released checkpoint has no EMA generator")
    generator_tensors = assign_state(
        generator, clean_state(state["generator_ema"]), device=device, dtype=torch.bfloat16
    )
    del state
    gc.collect()
    progress("loading_vae")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)
    pipeline = (
        ControlRollingForcingStreamingInferencePipeline(
            args,
            device=device,
            generator=generator,
            vae=vae,
            text_encoder=CachedTextEncoder(encoded).to(device),
        )
        .eval()
        .requires_grad_(False)
    )
    pipeline.vae_device = None
    config = DemoBackendConfig(
        config_path=str(repo / "configs/dmd_everything.yaml"),
        checkpoint_path=str(checkpoint),
        output_dir=str(repo / "outputs"),
        height=480,
        width=832,
        num_latent_frames=21,
        use_ema=True,
        device="cuda",
    )
    backend = InteractiveDemoBackend(config)
    backend.pipeline = pipeline
    backend.num_video_frames = 81
    attention = backend._rolling_forcing_attention_config()
    if not attention["enabled"]:
        raise RuntimeError("Rolling-force attention was not enabled")
    progress("model_ready")
    metadata = {
        "backend": "StreamForce released student EMA; external pretrained model",
        "source_revision": json.loads((repo / "SOURCE.json").read_text())["revision"],
        "loader": "strict meta initialization, mmap state, staged GPU text embeddings",
        "text_tensors": text_tensors,
        "generator_tensors": generator_tensors,
        "generator_parameters": sum(p.numel() for p in generator.parameters()),
        "dtype": "bfloat16",
        "attention": attention,
        "captioner": False,
        "numerical_parity_with_original_loading": (
            "not independently executed; all tensors loaded strictly"
        ),
    }
    return backend, metadata
