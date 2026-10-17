"""Training-only angular labels from a checked deterministic simulator replay."""

import hashlib
import time

import numpy as np

from click2future.data import load_rollout
from click2future.models.object_motion import encode_motion, observe
from click2future.sim.backend import apply_force, build_space, snapshot, step

from .train_cvae import dataset_index


def replay_angles(reference):
    scene = reference["scene_spec"]
    meta = reference["metadata"]
    state = build_space(scene, physics_hz=meta["physics_hz"])
    angles, positions = [], []
    for frame in range(meta["frame_count"]):
        if frame:
            for _ in range(meta["physics_hz"] // meta["fps"]):
                if state["time"] < reference["force_duration"]:
                    apply_force(
                        state,
                        reference["selected_object_id"],
                        tuple(reference["force_xy"].tolist()),
                    )
                step(state)
        objects = snapshot(state)["objects"]
        angles.append([obj["angle"] for obj in objects])
        positions.append([obj["position"] for obj in objects])
    if not np.allclose(positions, reference["positions"], atol=1e-6, rtol=1e-6):
        raise ValueError("Angular-label replay changed the frozen trajectory")
    return np.asarray(angles)


def canonical_ids(observation, selected_id):
    return (
        [selected_id, 1 - selected_id]
        if observation.selected_index == 0
        else [1 - selected_id, selected_id]
    )


def prepare_trajectories(root, *, augment=True):
    """Read train/validation only. Replay is label creation, never model inference."""
    started = time.perf_counter()
    selected, _, provenance = dataset_index(root)
    result, calibration = {}, {}
    label_digest = hashlib.sha256()
    for split, ids in selected.items():
        features, forces, motions, examples = [], [], [], []
        minimum_iou = 1.0
        for sample_id in ids:
            ref = load_rollout(root, sample_id)
            if (
                ref["frames"].shape != (32, 64, 64, 3)
                or len(ref["object_valid"]) != 2
                or not ref["object_valid"].all()
                or not np.isclose(ref["force_duration"], 0.25)
                or not np.allclose(np.diff(ref["timestamps"]), 0.125)
            ):
                raise ValueError(
                    "The trajectory pilot requires the frozen two-object 64px/32-frame clips"
                )
            initial = observe(ref["frames"][0], ref["selected_mask"])
            order = canonical_ids(initial, ref["selected_object_id"])
            truth_masks = ref["instance_masks"][0, order]
            iou = (initial.masks & truth_masks).sum((1, 2)) / (initial.masks | truth_masks).sum(
                (1, 2)
            )
            minimum_iou = min(minimum_iou, float(iou.min()))
            if iou.min() < 0.95:
                raise ValueError("Frame-0 extraction fails the declared calibration gate")
            angles = replay_angles(ref)
            label_digest.update(angles.tobytes())
            target = encode_motion(ref["positions"][:, order], angles[:, order])
            variants = (
                [(k, mirror) for k in range(4) for mirror in (False, True)]
                if augment and split == "train"
                else [(0, False)]
            )
            for quarter_turns, mirror in variants:
                image = np.rot90(ref["frames"][0], quarter_turns).copy()
                mask = np.rot90(ref["selected_mask"], quarter_turns).copy()
                matrix = np.linalg.matrix_power(np.array([[0, 1], [-1, 0]]), quarter_turns)
                if mirror:
                    image, mask = np.fliplr(image).copy(), np.fliplr(mask).copy()
                    matrix = np.diag([-1, 1]) @ matrix
                augmented = observe(image, mask)
                augmented_order = canonical_ids(augmented, ref["selected_object_id"])
                permutation = [order.index(obj) for obj in augmented_order]
                motion = target[:, permutation].copy()
                motion[..., :2] = motion[..., :2] @ matrix.T
                motion[..., 2] *= round(np.linalg.det(matrix))
                features.append(augmented.features)
                forces.append(matrix @ ref["force_xy"])
                motions.append(motion)
                examples.append(
                    {"sample_id": sample_id, "quarter_turns": quarter_turns, "mirror": mirror}
                )
        result[split] = {
            "features": np.asarray(features, dtype=np.float32),
            "force": np.asarray(forces, dtype=np.float32),
            "motion": np.asarray(motions, dtype=np.float32),
            "examples": examples,
        }
        calibration[split] = {
            "original_clips": len(ids),
            "training_examples": len(examples),
            "minimum_frame0_mask_iou": minimum_iou,
        }
    return result, {
        **provenance,
        "calibration": calibration,
        "angular_labels_sha256": label_digest.hexdigest(),
        "angular_label_policy": "training-only checked replay; original rollout files unchanged",
        "augmentation": "train-only square rotations/reflections" if augment else "none",
        "preparation_seconds": time.perf_counter() - started,
    }
