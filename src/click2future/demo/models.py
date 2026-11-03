"""Checkpoint adapters sharing the offline sampling equations and observable inputs."""

from pathlib import Path

import numpy as np
import torch

from click2future.models.cvae import ConditionalVideoVAE
from click2future.training.conditioning import observable_condition
from click2future.training.sample_diffusion import load_generator, sample_videos
from click2future.training.train_cvae import sha256, validate_config
from click2future.training.train_diffusion import TrainConfig


class CheckpointModel:
    def __init__(
        self,
        name,
        checkpoint,
        *,
        device="cpu",
        force_limit=0.5,
        sampling_steps=32,
        trajectory_steps=16,
    ):
        self.name, self.device = name, device
        checkpoint = Path(checkpoint)
        digest = sha256(checkpoint)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.sampling_steps = sampling_steps
        self.future_frames = None
        self.resolution = None
        if state.get("trajectory_format_version") in (1, 2, 3, 4, 5, 6, 7, 8):
            from click2future.training.sample_trajectory import load_trajectory

            self.model, state = load_trajectory(checkpoint, device)
            if not (
                2 <= trajectory_steps <= 128
                or (getattr(self.model, "world_prior_only", False) and trajectory_steps == 1)
            ):
                raise ValueError("Use 2-128 trajectory sampling steps")
            self.sampling_steps = (
                1 if getattr(self.model, "world_prior_only", False) else trajectory_steps
            )
            self.future_frames, self.resolution = self.model.future_frames, 64
            self.kind, step = "trajectory_diffusion", state["step"]
            force_limit = min(force_limit, self.model.force_limit)
            detail = "supervised object-trajectory diffusion, rigid observed-pixel decoder, EMA"
            if state["trajectory_format_version"] == 2:
                detail += "; contact/stationary training; unchanged inference"
            elif state["trajectory_format_version"] == 3:
                detail += "; experimental shared-world relational refinement"
            elif state["trajectory_format_version"] == 4:
                detail += "; experimental temporal residual on frozen E7"
            elif state["trajectory_format_version"] == 6:
                detail += "; force routing: " + self.model.force_routing
            elif state["trajectory_format_version"] in (7, 8):
                detail = (
                    "one-pass prior-conditioned pose decoder, rigid observed-pixel decoder, EMA"
                    if self.model.world_prior_only
                    else "conditional diffusion control for the prior-only comparison, EMA"
                )
                if state["trajectory_format_version"] == 8:
                    detail += "; D019 training-loss candidate; not quality-promoted"
            elif state["trajectory_format_version"] == 5:
                detail += "; experimental observed-surface refinement"
        elif state.get("checkpoint_version") == 1:
            config = TrainConfig(**state["config"])
            config.validate()
            if not 2 <= sampling_steps <= config.diffusion_steps:
                raise ValueError("Sampling steps exceed the saved diffusion schedule")
            with torch.random.fork_rng(devices=[]):
                self.model, self.diffusion, _ = load_generator(checkpoint, device)
            step = state["step"]
            self.kind = "diffusion"
            force_limit = min(force_limit, config.force_limit)
            detail = f"{config.conditioning} diffusion, {config.representation}, EMA"
        elif state.get("format_version") == 1:
            config = state["config"]
            validate_config(config)
            if state["model_config"] != config["model"]:
                raise ValueError("Checkpoint model and run configuration disagree")
            folder = Path(__file__).parents[1] / "models"
            sources = state["provenance"].get("source_sha256", {})
            if sources and sources.get("cvae.py") != sha256(folder / "cvae.py"):
                raise ValueError("CVAE model source differs from the checkpoint")
            if config["model"].get("decoder_kind") == "flow":
                if not (folder / "cvae_flow.py").exists():
                    raise ValueError("This checkpoint needs the reviewed flow decoder source")
                if sources.get("cvae_flow.py") != sha256(folder / "cvae_flow.py"):
                    raise ValueError("Flow decoder source differs from the checkpoint")
                from click2future.models.cvae_flow import build_cvae

                factory = build_cvae
            else:

                def factory(settings):
                    return ConditionalVideoVAE(**settings)

            with torch.random.fork_rng(devices=[]):
                self.model = factory(config["model"])
            self.model.load_state_dict(state["model"], strict=True)
            self.model.to(device).eval()
            self.force_scale = config["data"]["force_scale"]
            self.future_frames = config["model"]["future_frames"]
            self.resolution = config["data"]["resolution"]
            self.kind = "cvae"
            step = state["state"]["step"]
            detail = "conditional VAE, learned prior"
        else:
            raise ValueError("Use a supported diffusion or CVAE checkpoint")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError(
                "The demo requires a trained checkpoint; fixtures are not model results"
            )
        if sha256(checkpoint) != digest:
            raise ValueError("Checkpoint changed while loading")
        self.force_limit = force_limit
        self.metadata = {
            "id": name,
            "kind": self.kind,
            "detail": detail,
            "checkpoint_sha256": digest,
            "training_steps": step,
            "quality": "experimental checkpoint; held-out quality is not certified by this demo",
            "sampling_steps": self.sampling_steps
            if self.kind in {"diffusion", "trajectory_diffusion"}
            else None,
            "sampling_source_sha256": {
                str(path.relative_to(Path(__file__).parents[1])): sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).parents[1] / "training/diffusion.py",
                    Path(__file__).parents[1] / "training/sample_diffusion.py",
                    Path(__file__).parents[1] / "training/conditioning.py",
                    Path(__file__).parents[1] / "models/cvae.py",
                    Path(__file__).parents[1] / "models/pilot_denoiser.py",
                )
            },
            "runtime": {
                "torch": torch.__version__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name() if str(device).startswith("cuda") else None,
                "precision": "float32",
            },
        }
        flow_source = Path(__file__).parents[1] / "models/cvae_flow.py"
        if self.kind == "cvae" and flow_source.exists():
            self.metadata["sampling_source_sha256"]["models/cvae_flow.py"] = sha256(flow_source)
        if self.kind == "trajectory_diffusion":
            folder = Path(__file__).parents[1]
            for name in ("models/object_motion.py", "training/sample_trajectory.py"):
                self.metadata["sampling_source_sha256"][name] = sha256(folder / name)
            self.metadata.update(
                sampler="direct world-conditioned decode"
                if getattr(self.model, "world_prior_only", False)
                else "EDM rho-7 Heun, no churn, float32 integration",
                network_evaluations_per_sample=2 * self.sampling_steps - 1,
                supervision="future positions and checked angular replay; training only",
            )
            if state["trajectory_format_version"] == 2:
                self.metadata["training_variant"] = "contact_regularized"
                self.metadata["sampling_source_sha256"]["training/contact_checkpoint.py"] = sha256(
                    folder / "training/contact_checkpoint.py"
                )
            elif state["trajectory_format_version"] == 6:
                self.metadata.update(
                    training_variant="force_routing", force_routing=self.model.force_routing
                )
                for name in ("models/routed_motion.py", "training/routing_checkpoint.py"):
                    self.metadata["sampling_source_sha256"][name] = sha256(folder / name)
            elif state["trajectory_format_version"] in (3, 4, 5, 7, 8):
                self.metadata.update(
                    training_variant="shared_world_relational",
                    world_condition=state["config"]["world_condition"],
                    relational=state["config"]["relational"],
                )
                if state["config"]["world_condition"]:
                    self.metadata["supervision"] += (
                        "; physical labels in training, prior at inference"
                    )
                for name in ("models/world_motion.py", "training/world_checkpoint.py"):
                    self.metadata["sampling_source_sha256"][name] = sha256(folder / name)
                if state["trajectory_format_version"] in (7, 8):
                    self.metadata.update(
                        training_variant="prior_only_comparison",
                        world_prior_only=state["config"]["world_prior_only"],
                        prior_contract=state["prior_contract"],
                    )
                    for name in ("models/prior_motion.py", "training/prior_checkpoint.py"):
                        self.metadata["sampling_source_sha256"][name] = sha256(folder / name)
                    if state["trajectory_format_version"] == 8:
                        self.metadata.update(
                            training_variant="expected_direct_objective",
                            study_id="D019",
                            objective_contract=state["objective_contract"],
                            expected_objective=state["config"]["expected_objective"],
                        )
                elif state["trajectory_format_version"] == 4:
                    self.metadata.update(
                        training_variant="temporal_residual",
                        temporal_memory=state["config"]["temporal_memory"],
                        frozen_backbone_sha256=state["frozen_backbone_sha256"],
                    )
                    for name in ("models/contact_memory.py", "training/memory_checkpoint.py"):
                        self.metadata["sampling_source_sha256"][name] = sha256(folder / name)
                elif state["trajectory_format_version"] == 5:
                    self.metadata.update(
                        training_variant="surface_messages",
                        local_surfaces=state["config"]["local_surfaces"],
                        surface_contract=state["surface_contract"],
                    )
                    for name in (
                        "models/surface_motion.py",
                        "training/surface_checkpoint.py",
                        "training/contact_geometry.py",
                    ):
                        self.metadata["sampling_source_sha256"][name] = sha256(folder / name)

    @torch.inference_mode()
    def predict(self, first_frame, selected_mask, force_xy, *, seeds, future_frames, cancelled):
        if self.future_frames is not None and future_frames != self.future_frames:
            raise ValueError("Scene length differs from the checkpoint")
        if self.resolution is not None and first_frame.shape[:2] != (self.resolution,) * 2:
            raise ValueError("Scene resolution differs from the checkpoint")
        if self.kind == "trajectory_diffusion":
            from click2future.training.sample_trajectory import sample_trajectory_videos

            return sample_trajectory_videos(
                self.model,
                first_frame,
                selected_mask,
                force_xy,
                seeds=seeds,
                sampling_steps=self.sampling_steps,
                force_limit=self.force_limit,
                cancelled=cancelled,
            )
        videos = []
        for seed in seeds:
            if cancelled.is_set():
                raise InterruptedError("Generation cancelled")
            if self.kind == "diffusion":
                video = sample_videos(
                    self.model,
                    self.diffusion,
                    first_frame,
                    selected_mask,
                    force_xy,
                    seeds=(seed,),
                    future_frames=future_frames,
                    sampling_steps=self.sampling_steps,
                    force_limit=self.force_limit,
                )[0]
            else:
                # Match sample_cvae exactly, including float32 conversion order.
                first = torch.from_numpy(first_frame[None].transpose(0, 3, 1, 2).copy())
                first = first.to(self.device).float() / 127.5 - 1
                mask = torch.from_numpy(selected_mask[None, None].copy()).to(self.device)
                force = torch.from_numpy(force_xy[None].copy()).to(self.device)
                observable_condition(first / 2 + 0.5, mask, force, force_limit=self.force_limit)
                future = self.model.sample(first, mask, force / self.force_scale, seeds=[seed])[
                    0, 0
                ]
                if not torch.isfinite(future).all():
                    raise ValueError("Checkpoint returned nonfinite pixels")
                pixels = ((future.permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255)
                video = np.concatenate((first_frame[None], pixels.byte().cpu().numpy()))
            videos.append(video)
        return np.stack(videos)
