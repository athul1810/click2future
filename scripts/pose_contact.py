"""Contact timing from predicted poses, calibrated on simulator references.

The pixel contact estimator scores 0.489 event F1 on correct validation renders
(contact-reference-control.json), motivating a separate geometry-based timing
control. For models that save rigid poses, contact can be read from geometry.
This moves the scene's exact object
outlines (evaluation-only ground truth, never a model input) by the saved poses,
in the same frame-zero encoding as `rigid_geometry.geometry_diagnostics`, and
detects a contact onset in interval i when the outlines are separated at frame i
and touching at frame i+1.

Three parts:

1. Reference calibration on training/validation clips. Simulator poses are
   replayed exactly, encoded like model poses and scored against their own
   onset labels. The touch threshold is chosen on training clips only.
2. The diagnostic subset of labels satisfying this binary touch-edge rule.
3. Saved-pose evaluations: single-reference event F1, compared with the pixel
   audit, and a hidden-physics-aware first-contact timing measure, calibrated
   by scoring each reference against the other references of its condition.

No model, threshold or recorded score changes. Validation is the default scope.
Optional `--held-out` folders score a frozen held-out study (D018, `id_test`)
descriptively, with the touch threshold already chosen on training clips; no
selection, threshold or gate depends on them. `--held-out-study D028` reads the
D028 comparison layout instead: one folder per arm holding `<sample_id>.npz`
with `motion`, the shared `plan.json` and `results.json` one level up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from click2future.data import load_rollout
from click2future.evaluation.collision_audit import event_details, event_summary
from click2future.evaluation.rigid_geometry import _hull, _penetration
from click2future.models.object_motion import observe
from click2future.sim.backend import apply_force, build_space, snapshot, step
from click2future.sim.geometry import polygon_vertices

SPLITS = ("train", "validation")
TOUCH_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
VISIBLE_TOUCH = 0.05  # exact-shape gap treated as touching when classifying labels
TOLERANCE = 1  # the audit's one-interval event tolerance
HULL_TOUCH = 0.01  # pixel-hull gap read as held at contact (projection separates by 1e-3)
_REFERENCE_ONSETS = {}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay_poses(rollout):
    """Per-frame (x, y, angle) of each object from an exact simulator replay."""
    scene, meta = rollout["scene_spec"], rollout["metadata"]
    state = build_space(scene, physics_hz=meta["physics_hz"])
    selected = int(scene["action"]["selected_object_id"])
    force = tuple(scene["action"]["force_xy"])
    duration = float(scene["action"]["force_duration"])

    def grab():
        objects = sorted(snapshot(state)["objects"], key=lambda o: o["object_id"])
        return [(*o["position"], o["angle"]) for o in objects]

    poses = [grab()]
    for _ in range(meta["frame_count"] - 1):
        for _ in range(meta["physics_hz"] // meta["fps"]):
            if state["time"] < duration:
                apply_force(state, selected, force)
            step(state)
        poses.append(grab())
    poses = np.array(poses)
    error = float(np.abs(poses[:, :, :2] - rollout["positions"]).max())
    if error > 1e-6:
        raise ValueError(f"Replay differs from the saved positions by {error}")
    return poses


def observation_order(rollout):
    """Observation in the audit's object order, and the matching scene object IDs."""
    obs = observe(rollout["frames"][0], np.asarray(rollout["selected_mask"]).astype(np.uint8))
    masks = np.asarray(rollout["instance_masks"][0])
    order = [
        int(np.argmax([(masks[oid] & obs.masks[i]).sum() for oid in range(2)])) for i in range(2)
    ]
    if order[0] == order[1]:
        raise ValueError("Observed masks do not match two distinct scene objects")
    return obs, order


def outlines(scene, order, resolution, offset):
    """Exact convex frame-zero outlines in audit pixel coordinates (circles as 256-gons)."""
    result = []
    for oid in order:
        obj = next(o for o in scene["objects"] if o["object_id"] == oid)
        x, y = obj["position"]
        px, py = x * resolution + offset, y * resolution + offset
        if obj["shape"] == "circle":
            radius = obj["geometry"]["radius"] * resolution
            angles = np.linspace(0, 2 * math.pi, 256, endpoint=False)
            result.append(
                np.stack((px + radius * np.cos(angles), py + radius * np.sin(angles)), -1)
            )
        else:
            c, s = math.cos(obj["angle"]), math.sin(obj["angle"])
            vertices = polygon_vertices(obj["shape"], obj["geometry"])
            result.append(
                np.array(
                    [
                        [px + (lx * c - ly * s) * resolution, py + (lx * s + ly * c) * resolution]
                        for lx, ly in vertices
                    ]
                )
            )
    return result


def _points_to_edges(points, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab = b - a
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)[None], 0, 1)
    closest = a[None] + t[..., None] * ab[None]
    return float(np.sqrt(((points[:, None, :] - closest) ** 2).sum(-1)).min())


def signed_distance(first, second):
    """Gap between convex outlines in pixels, or minus the penetration depth."""
    depth = _penetration(first, second)
    if depth > 0:
        return -depth
    return min(_points_to_edges(first, second), _points_to_edges(second, first))


def move(polygons, centres, frame, resolution):
    """Apply one saved pose frame [2,4] exactly as `geometry_diagnostics` does."""
    moved = []
    for polygon, centre, value in zip(polygons, centres, frame, strict=True):
        dx, dy = value[:2] * 0.5 * resolution
        angle = math.atan2(value[2], value[3] * 2 + 1)
        c, s = math.cos(angle), math.sin(angle)
        moved.append((polygon - centre) @ np.array([[c, -s], [s, c]]).T + centre + [dx, dy])
    return moved


def encode(poses, centres, order, resolution, offset):
    """Encode simulator poses [T,N,3] as saved-pose motion [T-1,2,4]."""
    motion = np.zeros((poses.shape[0] - 1, 2, 4))
    for t in range(1, poses.shape[0]):
        for i, oid in enumerate(order):
            x0, y0, a0 = poses[0, oid]
            xt, yt, at = poses[t, oid]
            c, s = math.cos(at - a0), math.sin(at - a0)
            start = np.array([x0, y0]) * resolution + offset
            now = np.array([xt, yt]) * resolution + offset
            shift = now + np.array([[c, -s], [s, c]]) @ (centres[i] - start) - centres[i]
            motion[t - 1, i] = (*(shift / (0.5 * resolution)), s, (c - 1) / 2)
    return motion


def distances(polygons, centres, motion, resolution):
    frames = [polygons] + [move(polygons, centres, frame, resolution) for frame in motion]
    return np.array([signed_distance(*pair) for pair in frames])


def onsets(distance, touch):
    contact = np.asarray(distance) <= touch
    return ~contact[:-1] & contact[1:]


def labels(rollout):
    onset = np.asarray(rollout["collision_onset"])
    return (onset[:, 0, 1] | onset[:, 1, 0]).astype(bool)


def summary(predictions, targets):
    rows = [
        event_details(p, y, np.ones_like(y), tolerance=TOLERANCE)
        for p, y in zip(predictions, targets, strict=True)
    ]
    return event_summary(rows)


def first_contact(onset):
    index = np.flatnonzero(onset)
    return int(index[0]) if len(index) else None


def first_contact_stats(pairs):
    """pairs of (candidate first-contact interval, reference first-contact intervals)."""
    occurrence, inside, error = [], [], []
    for candidate, references in pairs:
        if not references:
            continue  # A lone reference has no leave-one-out comparator.
        meet = [r for r in references if r is not None]
        if len(meet) in (0, len(references)):  # references agree on whether contact occurs
            occurrence.append((candidate is not None) == bool(meet))
        if candidate is not None and meet:
            inside.append(min(meet) - TOLERANCE <= candidate <= max(meet) + TOLERANCE)
            error.append(min(abs(candidate - r) for r in meet))
    return {
        "samples": len(pairs),
        "occurrence_scored": len(occurrence),
        "occurrence_agreement": float(np.mean(occurrence)) if occurrence else None,
        "timed": len(error),
        "within_reference_range_plus_minus_one_interval": float(np.mean(inside))
        if inside
        else None,
        "median_nearest_reference_error_intervals": float(np.median(error)) if error else None,
        "mean_nearest_reference_error_intervals": float(np.mean(error)) if error else None,
    }


def reference_geometry(rollout, offset):
    obs, order = observation_order(rollout)
    resolution = rollout["frames"].shape[1]
    centres = [obs.centers[i] * resolution for i in range(2)]
    polygons = outlines(rollout["scene_spec"], order, resolution, offset)
    return polygons, centres, order, resolution


def reference_distance(rollout, offset):
    poses = replay_poses(rollout)
    polygons, centres, order, resolution = reference_geometry(rollout, offset)
    return distances(
        polygons, centres, encode(poses, centres, order, resolution, offset), resolution
    )


def calibrate(dataset, offset):
    manifest = json.loads((Path(dataset) / "dataset_manifest.json").read_text())
    clips = {split: [] for split in SPLITS}
    for row in sorted(manifest["samples"], key=lambda r: r["sample_id"]):
        if row["split"] in SPLITS:
            rollout = load_rollout(dataset, row["sample_id"])
            clips[row["split"]].append((reference_distance(rollout, offset), labels(rollout)))
    train = {
        f"{touch:g}": summary(
            [onsets(d, touch) for d, _ in clips["train"]], [y for _, y in clips["train"]]
        )["f1"]
        for touch in TOUCH_GRID
    }
    touch = float(max(TOUCH_GRID, key=lambda t: train[f"{t:g}"]))
    result = {
        "offset_pixels": offset,
        "touch_selected_on_train_pixels": touch,
        "train_f1_by_touch": train,
        "visible_subset_scope": (
            "Labels intersected with the fixed VISIBLE_TOUCH gap-to-touch rule; "
            "a rule-conditioned diagnostic, not independent detector validation."
        ),
    }
    for split, rows in clips.items():
        distance = [d for d, _ in rows]
        target = [y for _, y in rows]
        visible = [y & (d[:-1] > VISIBLE_TOUCH) & (d[1:] <= VISIBLE_TOUCH) for d, y in rows]
        repeat = 0
        for d, y in rows:
            index = np.flatnonzero(y)
            repeat += sum(
                1
                for a, b in zip(index[:-1], index[1:], strict=True)
                if not (d[a + 1 : b + 1] > VISIBLE_TOUCH).any()
            )
        predicted = [onsets(d, touch) for d in distance]
        result[split] = {
            "clips": len(rows),
            "labelled_onsets": int(sum(y.sum() for y in target)),
            "frame_visible_onsets": int(sum(v.sum() for v in visible)),
            "already_touching_at_interval_start": int(
                sum((y & (d[:-1] <= VISIBLE_TOUCH)).sum() for d, y in rows)
            ),
            "repeat_onsets_without_visible_separation": repeat,
            "events_vs_all_labels": summary(predicted, target),
            "events_vs_frame_visible_labels": summary(predicted, visible),
        }
    return result


def reference_onsets(dataset, name, touch, offset):
    """Onsets of one replayed reference, cached across arms that share references."""
    key = (str(Path(dataset).resolve()), name, touch, offset)
    if key not in _REFERENCE_ONSETS:
        _REFERENCE_ONSETS[key] = onsets(
            reference_distance(load_rollout(dataset, name), offset), touch
        )
    return _REFERENCE_ONSETS[key]


def _pixel_events(folder, flat):
    """Pixel-audit events against the first reference, from either result layout."""
    if flat:  # D028: counts per arm in the shared results.json
        counts = json.loads((folder.parent / "results.json").read_text())["models"][folder.name][
            "contact_counts"
        ]
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        return {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        }
    audit = folder / "collision-audit.json"
    if not audit.exists():
        audit = folder / "collision.json"  # D018 file name
    pixel = json.loads(audit.read_text())
    return {k: v for k, v in pixel["model_events_against_first_reference"].items() if k != "scope"}


def evaluate(folder, dataset, touch, offset, split="validation"):
    folder = Path(folder)
    flat = not (folder / "geometry-audit").is_dir()
    plan_path = folder / "plan.json"
    if flat and not plan_path.exists():
        plan_path = folder.parent / "plan.json"
    plan = json.loads(plan_path.read_text())
    if plan["split"] != split:
        raise ValueError(f"Expected a {split} plan, found {plan['split']}")
    first_rows, all_rows, alternative, leave_one_out, timing = [], [], [], [], []
    pose_hash = hashlib.sha256()
    for case in plan["cases"]:
        references = [load_rollout(dataset, name) for name in case["reference_ids"]]
        base = references[0]
        polygons, centres, _, resolution = reference_geometry(base, offset)
        reference_first = []
        for name, reference in zip(case["reference_ids"], references, strict=True):
            onset = reference_onsets(dataset, name, touch, offset)
            reference_first.append(first_contact(onset))
            if reference is not base:
                alternative.append(
                    event_details(
                        onset, labels(base), np.ones(onset.size, bool), tolerance=TOLERANCE
                    )
                )
        leave_one_out += [
            (reference_first[i], reference_first[:i] + reference_first[i + 1 :])
            for i in range(len(references))
        ]
        path = (
            folder / f"{case['sample_id']}.npz"
            if flat
            else folder / "geometry-audit" / f"{case['sample_id']}-poses.npz"
        )
        pose_hash.update(bytes.fromhex(sha256(path)))
        with np.load(path, allow_pickle=False) as archive:
            motions = archive["motion"]
        for motion in motions:
            onset = onsets(distances(polygons, centres, motion, resolution), touch)
            first_rows.append(
                event_details(onset, labels(base), np.ones(onset.size, bool), tolerance=TOLERANCE)
            )
            all_rows += [
                event_details(onset, labels(r), np.ones(onset.size, bool), tolerance=TOLERANCE)
                for r in references
            ]
            timing.append((first_contact(onset), reference_first))
    return {
        "plan_sha256": plan["plan_sha256"],
        "poses_sha256": pose_hash.hexdigest(),
        "pixel_events_vs_first_reference": _pixel_events(folder, flat),
        "pose_events_vs_first_reference": event_summary(first_rows),
        "pose_events_vs_each_reference": event_summary(all_rows),
        "first_contact": first_contact_stats(timing),
        "alternative_physics_events_vs_first_reference": event_summary(alternative),
        "reference_first_contact_leave_one_out": first_contact_stats(leave_one_out),
    }


def standoff(folders, dataset, touch, offset):
    """Diagnostic for hull-based corrections: exact gaps where pixel-centre hulls meet.

    Records, for contact conditions, the pixel-centre hull gap at frames where the
    replayed references truly touch, each arm's closest exact gap, and, for samples
    that never reach exact contact, the pixel-hull gap at their closest exact frame.
    """
    plan = json.loads((Path(next(iter(folders.values()))).parent / "plan.json").read_text())
    hull_gap, closest = [], {name: [] for name in folders}
    missed = {name: [] for name in folders}
    for case in plan["cases"]:
        names = case["reference_ids"]
        if all(first_contact(reference_onsets(dataset, n, touch, offset)) is None for n in names):
            continue
        base = load_rollout(dataset, names[0])
        polygons, centres, order, resolution = reference_geometry(base, offset)
        obs, _ = observation_order(base)
        hulls = [_hull(np.stack((x + 0.5, y + 0.5), -1)) for y, x in map(np.nonzero, obs.masks)]
        for name in names:
            reference = load_rollout(dataset, name)
            motion = encode(replay_poses(reference), centres, order, resolution, offset)
            exact = distances(polygons, centres, motion, resolution)
            for t in np.flatnonzero(exact[1:] <= touch):
                hull_gap.append(signed_distance(*move(hulls, centres, motion[t], resolution)))
        for arm, folder in folders.items():
            with np.load(Path(folder) / f"{case['sample_id']}.npz", allow_pickle=False) as a:
                for motion in a["motion"]:
                    exact = distances(polygons, centres, motion, resolution)
                    t = 1 + int(np.argmin(exact[1:]))
                    closest[arm].append(float(exact[t]))
                    if first_contact(onsets(exact, touch)) is None:
                        frame = motion[t - 1]
                        gap = signed_distance(*move(hulls, centres, frame, resolution))
                        missed[arm].append((float(exact[t]), float(gap)))

    def quantiles(values):
        q = np.percentile(values, [5, 25, 50, 75, 95])
        return dict(zip(("p5", "p25", "median", "p75", "p95"), q.tolist(), strict=True))

    def misses(rows):
        held = [e for e, h in rows if abs(h) <= HULL_TOUCH]
        apart = [h for e, h in rows if abs(h) > HULL_TOUCH]
        return {
            "samples_without_exact_contact": len(rows),
            "pixel_hulls_touching_at_closest_frame": len(held),
            "exact_gap_when_hulls_touching_pixels": quantiles(held) if held else None,
            "pixel_hull_gap_otherwise_pixels": quantiles(apart) if apart else None,
        }

    return {
        "scope": "Diagnostic over contact conditions; changes no score.",
        "reference_touching_frames": len(hull_gap),
        "pixel_hull_gap_at_true_contact_pixels": quantiles(hull_gap),
        "reference_touching_frames_with_overlapping_hulls": int(np.sum(np.array(hull_gap) < 0)),
        "closest_exact_gap_pixels": {
            arm: {"samples": len(v), **quantiles(v)} for arm, v in closest.items()
        },
        "hull_touch_tolerance_pixels": HULL_TOUCH,
        "samples_without_exact_contact": {arm: misses(v) for arm, v in missed.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--evaluation", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--held-out-dataset", type=Path)
    parser.add_argument("--held-out", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--held-out-study", default="D018", choices=("D018", "D028"))
    parser.add_argument(
        "--held-out-touch",
        type=float,
        nargs="*",
        default=[],
        help="extra touch thresholds (pixels) for a labelled held-out sensitivity check",
    )
    parser.add_argument(
        "--standoff",
        action="store_true",
        help="record the hull-standoff diagnostic for D028-layout held-out arms",
    )
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.2, 0.0, 0.5])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Preserve earlier results; use a fresh output path")
    evaluations = [item.partition("=")[::2] for item in args.evaluation]
    held_out = [item.partition("=")[::2] for item in args.held_out]
    if any(not name or not folder for name, folder in evaluations + held_out):
        parser.error("Use NAME=FOLDER")
    if bool(held_out) != bool(args.held_out_dataset):
        parser.error("--held-out needs --held-out-dataset, and the reverse")
    if args.standoff and (not held_out or args.held_out_study != "D028"):
        parser.error("--standoff needs D028-layout --held-out arms")
    runs = []
    for offset in args.offsets:
        calibration = calibrate(args.dataset, offset)
        touch = calibration["touch_selected_on_train_pixels"]
        runs.append(
            {
                "calibration": calibration,
                "evaluations": {
                    name: evaluate(folder, args.dataset, touch, offset)
                    for name, folder in evaluations
                },
                "held_out_id_test": {
                    name: evaluate(folder, args.held_out_dataset, touch, offset, "id_test")
                    for name, folder in held_out
                },
            }
        )
    record = {
        "status": "measured_pose_contact_control",
        "scope": (
            "Contact onsets from saved rigid poses with exact scene outlines, calibrated on "
            "training simulator references and checked on validation; original model, "
            "scorer thresholds and historical scores unchanged. Optional frozen "
            f"{args.held_out_study} id_test is descriptive only, with no selection or gate "
            "changes."
        ),
        "onset_rule": (
            "interval i: outlines separated at frame i and within the touch threshold at frame i+1"
        ),
        "event_tolerance_intervals": TOLERANCE,
        "pixel_reference_control_f1": 0.4888888888888889,
        "dataset_manifest_sha256": sha256(args.dataset / "dataset_manifest.json"),
        "held_out_dataset_manifest_sha256": sha256(args.held_out_dataset / "dataset_manifest.json")
        if held_out
        else None,
        "held_out_scope": (
            f"Descriptive only: frozen {args.held_out_study} checkpoints and plans, touch "
            "threshold chosen on training clips before reading them; no selection or gate."
        )
        if held_out
        else None,
        "source_sha256": {"pose_contact.py": sha256(__file__)},
        "primary_offset_pixels": args.offsets[0],
        "runs": runs,
        "held_out_touch_sensitivity": {
            "scope": (
                "Sensitivity only, at the primary offset: references and arms rescored at "
                "each touch threshold; the training-selected threshold stays primary."
            ),
            "thresholds": {
                f"{touch:g}": {
                    name: evaluate(folder, args.held_out_dataset, touch, args.offsets[0], "id_test")
                    for name, folder in held_out
                }
                for touch in args.held_out_touch
            },
        }
        if held_out and args.held_out_touch
        else None,
        "test_splits_read": ["id_test"] if held_out else [],
        "limitations": [
            "Exact outlines come from the scene specification: evaluation-only ground "
            "truth, as labels are.",
            "Circles and rectangles only; triangles occur only in the shape-OOD split.",
            "This binary touch-edge rule does not identify onsets without a sampled "
            "gap-to-touch transition.",
            "Timing is conditional on detected contact; occurrence and timed denominators "
            "remain reported.",
            "Model samples use four references; leave-one-out reference comparisons use three.",
            "A canonical zero-force reference has no leave-one-out peer and is unscorable "
            "for reference occurrence.",
            "Validation first-contact timing uses eight collision conditions; held-out "
            "D018 adds twenty on ten fresh ID scenes.",
        ],
    }
    if args.standoff:
        record["projection_standoff_diagnostic"] = standoff(
            dict(held_out),
            args.held_out_dataset,
            runs[0]["calibration"]["touch_selected_on_train_pixels"],
            args.offsets[0],
        )
    if record["held_out_touch_sensitivity"] is None:
        del record["held_out_touch_sensitivity"]  # keep the default record unchanged
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    primary = runs[0]
    calibration = primary["calibration"]
    print(
        json.dumps(
            {
                "touch": calibration["touch_selected_on_train_pixels"],
                "validation": {
                    k: calibration["validation"][k]
                    for k in calibration["validation"]
                    if not k.startswith("events")
                },
                "validation_f1_all": calibration["validation"]["events_vs_all_labels"]["f1"],
                "validation_f1_visible": calibration["validation"][
                    "events_vs_frame_visible_labels"
                ]["f1"],
            },
            indent=1,
        )
    )
    for name, e in primary["evaluations"].items():
        print(
            name,
            "pixel F1",
            round(e["pixel_events_vs_first_reference"]["f1"], 3),
            "| pose F1",
            round(e["pose_events_vs_first_reference"]["f1"], 3),
            "| first contact",
            {
                k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in e["first_contact"].items()
            },
        )


if __name__ == "__main__":
    main()
