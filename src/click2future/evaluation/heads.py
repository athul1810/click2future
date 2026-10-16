"""Export denoiser heads in simulator ID order for separate auxiliary-head scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from click2future.data import validate_rollout
from click2future.models.pilot_denoiser import DenoiserOutput
from click2future.training.targets import canonical_object_order, to_scene_order

from .metrics import collision_metrics, trajectory_metrics


class CaptureFinalHeads:
    """Retain timestep-zero heads from one sampler call without changing its output.

    Works with the existing denoiser call interface, including B's DDIM sampler.
    Only small detached head tensors are copied; noisy video features are not kept.
    """

    def __init__(self, model):
        self.model = model
        self._handle = None
        self._output = None
        self._last_timestep = None

    def __enter__(self):
        if self._handle is not None:
            raise ValueError("Head capture cannot be nested")
        self._output = self._last_timestep = None
        self._handle = self.model.register_forward_hook(self._capture, with_kwargs=True)
        return self

    def _capture(self, model, args, kwargs, output):
        times = args[1] if len(args) > 1 else kwargs["timesteps"]
        if times.ndim != 1 or not len(times) or not (times == times[0]).all():
            raise ValueError("Sampler head capture requires one common timestep per batch")
        timestep = int(times[0])
        if timestep < 0 or (self._last_timestep is not None and timestep > self._last_timestep):
            raise ValueError("Use one capture context per decreasing-timestep sampler call")
        self._last_timestep = timestep
        if timestep == 0:
            if output.positions is None or output.collision_logits is None:
                raise ValueError("Head capture requires an auxiliary-enabled checkpoint")
            self._output = DenoiserOutput(
                torch.empty(0),
                output.positions.detach().float().cpu().clone(),
                output.collision_logits.detach().float().cpu().clone(),
            )

    def __exit__(self, *error):
        self._handle.remove()
        self._handle = None

    @property
    def output(self):
        if self._output is None or self._last_timestep != 0:
            raise ValueError("The sampler must finish with a timestep-zero model call")
        return self._output


def evaluate_heads(
    reference: dict,
    outputs: list[DenoiserOutput],
    *,
    timesteps: list[int],
    num_diffusion_steps: int,
    max_timestep_fraction: float,
    collision_threshold: float,
    event_tolerance: int,
    run_id: str,
) -> tuple[dict, dict]:
    """Outputs are batch-size-one predictions from K sampler runs, not label-conditioned calls.

    The caller supplies the timestep of the actual denoiser features. Ground truth is
    used only after predictions exist; mapping uses frame-0 identity masks alone.
    This does not establish agreement between the head and generated RGB.
    """
    validate_rollout(reference)
    if (
        not outputs
        or len(outputs) != len(timesteps)
        or isinstance(num_diffusion_steps, bool)
        or not isinstance(num_diffusion_steps, int)
        or num_diffusion_steps < 2
        or not 0 <= max_timestep_fraction <= 1
        or not 0 < collision_threshold < 1
    ):
        raise ValueError("Provide K outputs/timesteps and valid schedule/threshold settings")
    cutoff = int(max_timestep_fraction * (num_diffusion_steps - 1))
    if any(
        isinstance(t, bool) or not isinstance(t, int) or not 0 <= t <= cutoff for t in timesteps
    ):
        raise ValueError("Head features must come from the declared supervised timestep range")
    frames, objects = reference["positions"].shape[:2]
    positions, logits = [], []
    for output in outputs:
        p, c = output.positions, output.collision_logits
        if p is None or c is None or p.ndim != 4 or p.shape[:2] != (1, frames - 1):
            raise ValueError("Every output needs batch-one heads aligned with future frames")
        p, c = p.detach().cpu(), c.detach().cpu()
        mapping = canonical_object_order(
            torch.from_numpy(reference["instance_masks"][:1]),
            torch.from_numpy(reference["object_valid"][None]),
            max_objects=p.shape[2],
        )
        restored, collision, _ = to_scene_order(p, c, mapping, scene_objects=objects)
        positions.append(restored[0].numpy())
        logits.append(collision[0].numpy())
    predicted = np.concatenate(
        (
            np.broadcast_to(reference["positions"][:1], (len(outputs), 1, objects, 2)),
            np.stack(positions),
        ),
        axis=1,
    )
    known = np.isfinite(predicted).all(-1) & reference["object_valid"][None, None]
    collision = np.stack(logits)
    observed = np.isfinite(collision) & known[:, :-1, :, None] & known[:, 1:, None, :]
    observed &= observed.swapaxes(-1, -2)
    onset = torch.from_numpy(collision).sigmoid().numpy() >= collision_threshold
    for array in (onset, observed):
        array[:, :, np.arange(objects), np.arange(objects)] = False
    arrays = {
        "predicted_positions": predicted,
        "reference_positions": reference["positions"],
        "object_valid": reference["object_valid"],
        "tracked": known,
        "predicted_collision_onset": onset,
        "reference_collision_onset": reference["collision_onset"],
        "collision_observed": observed,
        "timestamps": reference["timestamps"],
    }
    report = {
        "status": "passed",
        "metric_version": "1",
        "source": "auxiliary_head",
        "extractor_version": None,
        "run_id": run_id,
        "scene_group_id": reference["scene_spec"]["scene_group_id"],
        "condition_id": reference["scene_spec"]["condition_id"],
        "timestamps_seconds": reference["timestamps"].tolist(),
        "settings": {
            "head_timesteps": timesteps,
            "num_diffusion_steps": num_diffusion_steps,
            "max_timestep_fraction": max_timestep_fraction,
            "collision_threshold": collision_threshold,
            "matching": "fixed frame-0 mask-centre x/y order restored to scene IDs",
        },
        "trajectory": trajectory_metrics(
            predicted,
            reference["positions"],
            reference["object_valid"],
            known,
            selected_object=reference["selected_object_id"],
        ),
        "collision": collision_metrics(
            onset,
            reference["collision_onset"],
            reference["object_valid"],
            observed,
            tolerance=event_tolerance,
        ),
    }
    return report, arrays


def save_head_evaluation(
    report: dict, arrays: dict, *, sample_id: str, archive_path: Path, report_path: Path
):
    if (
        archive_path.resolve() == report_path.resolve()
        or archive_path.exists()
        or report_path.exists()
    ):
        raise ValueError("Use distinct new paths for the prediction archive and JSON report")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    result = {
        **report,
        "sample_id": sample_id,
        "input_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result
