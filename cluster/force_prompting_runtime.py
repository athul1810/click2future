"""Released point-force inference, with staged text encoding and strict weights."""

import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from streamforce_runtime import assign_state


class ForcePromptingBackend:
    def __init__(self, repo, *, progress=lambda *_: None):
        import torch
        from accelerate import init_empty_weights

        self.repo = Path(repo).resolve()
        self.base = self.repo / "weights/CogVideoX-5b-I2V"
        self.progress = progress
        self.embeddings = {}
        os.chdir(self.repo)
        sys.path.insert(0, str(self.repo / "src/force-prompting"))
        from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
        from models.cogvideo_controlnet import CogVideoXControlnet
        from models.cogvideo_transformer import CustomCogVideoXTransformer3DModel
        from pipelines.controlnet_img2vid_pipeline import CogVideoXImageToVideoControlnetPipeline
        from transformers import AutoTokenizer

        torch.set_grad_enabled(False)
        self.progress("loading_video_generator")
        transformer = (
            CustomCogVideoXTransformer3DModel.from_pretrained(
                self.base,
                subfolder="transformer",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to("cuda")
        )
        transformer.low_memory_mode = False
        with init_empty_weights():
            controlnet = CogVideoXControlnet(
                num_layers=6, downscale_coef=8, in_channels=3, num_attention_heads=48
            )
        checkpoint = self.repo / "checkpoints/step-5000-checkpoint-point-force.pt"
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        control_tensors = assign_state(
            controlnet, state["state_dict"], device="cuda", dtype=torch.bfloat16
        )
        del state
        controlnet.eval().requires_grad_(False)
        self.progress("loading_vae")
        vae = (
            AutoencoderKLCogVideoX.from_pretrained(
                self.base,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to("cuda")
        )
        vae.enable_slicing()
        vae.enable_tiling()
        tokenizer = AutoTokenizer.from_pretrained(
            self.base, subfolder="tokenizer", local_files_only=True
        )
        scheduler = CogVideoXDPMScheduler.from_pretrained(
            self.base, subfolder="scheduler", local_files_only=True
        )
        self.pipeline = CogVideoXImageToVideoControlnetPipeline(
            tokenizer=tokenizer,
            text_encoder=None,
            transformer=transformer,
            vae=vae,
            controlnet=controlnet,
            scheduler=scheduler,
        ).to("cuda")
        self.metadata = {
            "backend": "Force Prompting released point-force checkpoint; external pretrained model",
            "source_revision": json.loads((self.repo / "SOURCE.json").read_text())["revision"],
            "control_tensors": control_tensors,
            "dtype": "bfloat16",
            "steps": 50,
            "guidance_scale": 6.0,
            "frames": 49,
            "fps": 8,
            "size": [720, 480],
            "sampler": "upstream custom pipeline and CogVideoX DPM scheduler",
            "captioner": False,
            "low_memory_layer_truncation": False,
            "zero_semantics": "minimum native control; not zero physical force",
        }
        gc.collect()
        torch.cuda.empty_cache()
        self.progress("model_ready")

    def prepare_prompt(self, prompt):
        import torch
        from transformers import T5EncoderModel
        from utils.model_utils import _get_t5_prompt_embeds

        if prompt in self.embeddings:
            return
        self.progress("encoding_caption")
        encoder = (
            T5EncoderModel.from_pretrained(
                self.base,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to("cuda")
        )
        for text in dict.fromkeys([prompt, ""]):
            if text not in self.embeddings:
                self.embeddings[text] = (
                    _get_t5_prompt_embeds(
                        self.pipeline.tokenizer,
                        encoder,
                        text,
                        device="cuda",
                        dtype=torch.bfloat16,
                        max_sequence_length=226,
                    )
                    .detach()
                    .cpu()
                )
        del encoder
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, image, prompt, signal, seed, *, on_step=lambda *_: None):
        import numpy as np
        import torch

        self.prepare_prompt(prompt)
        self.progress("sampling")
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1

        def step(pipe, index, timestep, values):
            on_step(index + 1, 50)
            return values

        generated = self.pipeline(
            image=image_tensor.to("cuda"),
            controlnet_frames=signal[None].to("cuda"),
            prompt_embeds=self.embeddings[prompt].to("cuda"),
            negative_prompt_embeds=self.embeddings[""].to("cuda"),
            height=480,
            width=720,
            num_frames=49,
            num_inference_steps=50,
            guidance_scale=6.0,
            use_dynamic_cfg=False,
            controlnet_weights=1.0,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            output_type="np",
            callback_on_step_end=step,
        ).frames[0]
        generated = np.asarray(generated)
        if not np.isfinite(generated).all():
            raise FloatingPointError("The video generator produced non-finite pixels")
        return (generated * 255).clip(0, 255).astype(np.uint8)

    @staticmethod
    def signal(force):
        from data.controlnet_datasets import ForcePromptingDataset_PointForce

        # Retain the exact published signal builder, including its nonzero
        # minimum displacement. It is not a calibrated Newton-valued force.
        context = ForcePromptingDataset_PointForce.__new__(ForcePromptingDataset_PointForce)
        context.min_force = 0.0
        context.max_force = 1.0
        values = SimpleNamespace(**force["model"])
        return context.load_controlnet_signal(
            values.force,
            values.angle,
            values.x_pos,
            values.y_pos,
            num_frames=49,
            height=480,
            width=720,
        )
