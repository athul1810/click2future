"""Headless RGB and instance-mask rendering for simulator snapshots."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


DEFAULT_BACKGROUND = (210, 214, 220)
DEFAULT_BORDER = (70, 75, 82)


def render_rgb(
    scene: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    resolution: int = 64,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
    border_color: tuple[int, int, int] = DEFAULT_BORDER,
) -> np.ndarray:
    """Render one RGB frame as ``uint8[H,W,3]``."""

    _validate_resolution(resolution)
    image = Image.new("RGB", (resolution, resolution), color=background)
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (0, 0, resolution - 1, resolution - 1),
        outline=border_color,
        width=1,
    )
    _draw_objects(scene, snapshot, draw, resolution, masks=None)
    return np.asarray(image, dtype=np.uint8)


def render_instance_masks(
    scene: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    resolution: int = 64,
) -> np.ndarray:
    """Render binary object masks as ``bool[N,H,W]`` in object-id order."""

    _validate_resolution(resolution)
    object_ids = [int(obj["object_id"]) for obj in scene["objects"]]
    masks = np.zeros((len(object_ids), resolution, resolution), dtype=bool)
    mask_images = [Image.new("1", (resolution, resolution), color=0) for _ in object_ids]
    mask_draws = [ImageDraw.Draw(image) for image in mask_images]
    mask_index = {object_id: index for index, object_id in enumerate(object_ids)}
    _draw_objects(scene, snapshot, None, resolution, masks=(mask_index, mask_draws))

    for index, image in enumerate(mask_images):
        masks[index] = np.asarray(image, dtype=bool)
    return masks


def render_frame(
    scene: dict[str, Any],
    snapshot: dict[str, Any],
    *,
    resolution: int = 64,
    background: tuple[int, int, int] = DEFAULT_BACKGROUND,
) -> tuple[np.ndarray, np.ndarray]:
    """Render RGB and instance masks from the same snapshot."""

    return (
        render_rgb(scene, snapshot, resolution=resolution, background=background),
        render_instance_masks(scene, snapshot, resolution=resolution),
    )


def _draw_objects(
    scene: dict[str, Any],
    snapshot: dict[str, Any],
    draw: ImageDraw.ImageDraw | None,
    resolution: int,
    masks: tuple[dict[int, int], list[ImageDraw.ImageDraw]] | None,
) -> None:
    poses = {int(obj["object_id"]): obj for obj in snapshot["objects"]}
    for scene_object in sorted(scene["objects"], key=lambda value: value["object_id"]):
        object_id = int(scene_object["object_id"])
        pose = poses[object_id]
        points_or_box = _shape_pixels(scene, scene_object, pose, resolution)
        color = tuple(scene_object["appearance"]["color"])

        if scene_object["shape"] == "circle":
            if draw is not None:
                draw.ellipse(points_or_box, fill=color)
            _draw_mask(masks, object_id, "ellipse", points_or_box)
        elif scene_object["shape"] == "rectangle":
            if draw is not None:
                draw.polygon(points_or_box, fill=color)
            _draw_mask(masks, object_id, "polygon", points_or_box)
        else:
            raise ValueError(f"unsupported shape: {scene_object['shape']}")


def _shape_pixels(
    scene: dict[str, Any],
    scene_object: dict[str, Any],
    pose: dict[str, Any],
    resolution: int,
) -> tuple[float, ...] | list[tuple[float, float]]:
    world_width, world_height = scene["world_size"]
    x, y = pose["position"]
    px = float(x) / float(world_width) * resolution
    py = float(y) / float(world_height) * resolution
    angle = float(pose["angle"])
    geometry = scene_object["geometry"]

    if scene_object["shape"] == "circle":
        radius = float(geometry["radius"])
        rx = radius / float(world_width) * resolution
        ry = radius / float(world_height) * resolution
        return (px - rx, py - ry, px + rx, py + ry)

    width = float(geometry["width"])
    height = float(geometry["height"])
    local_points = (
        (-width / 2.0, -height / 2.0),
        (width / 2.0, -height / 2.0),
        (width / 2.0, height / 2.0),
        (-width / 2.0, height / 2.0),
    )
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return [
        (
            px + (local_x * cos_angle - local_y * sin_angle) / float(world_width) * resolution,
            py + (local_x * sin_angle + local_y * cos_angle) / float(world_height) * resolution,
        )
        for local_x, local_y in local_points
    ]


def _draw_mask(
    masks: tuple[dict[int, int], list[ImageDraw.ImageDraw]] | None,
    object_id: int,
    kind: str,
    geometry: tuple[float, ...] | list[tuple[float, float]],
) -> None:
    if masks is None:
        return
    mask_index, mask_draws = masks
    drawer = mask_draws[mask_index[object_id]]
    if kind == "ellipse":
        drawer.ellipse(geometry, fill=1)
    else:
        drawer.polygon(geometry, fill=1)


def _validate_resolution(resolution: int) -> None:
    if resolution <= 0:
        raise ValueError("resolution must be positive")
