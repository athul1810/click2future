"""Read supported tabletop pixels without a scene recipe or physical labels."""

import base64
import binascii
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from click2future.models.object_motion import _components, observe

MAX_IMAGE_BYTES = 65536
IMAGE_PREFIX = "data:image/png;base64,"


def uploaded_scene(value):
    if not isinstance(value, str) or not value.startswith(IMAGE_PREFIX) or len(value) > 90000:
        raise ValueError("Open a 64 x 64 synthetic tabletop PNG, at most 64 KiB")
    try:
        payload = base64.b64decode(value[len(IMAGE_PREFIX) :], validate=True)
        if not payload or len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("PNG exceeds the 64 KiB limit")
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.size != (64, 64):
                raise ValueError("Use a native 64 x 64 PNG; images are never resized")
            if getattr(image, "n_frames", 1) != 1 or image.mode not in {"RGB", "RGBA", "P"}:
                raise ValueError("Use one opaque colour PNG frame")
            image.load()
            if np.any(np.asarray(image.convert("RGBA"))[..., 3] != 255):
                raise ValueError("Transparent images are unsupported")
            first = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    except (
        binascii.Error,
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        EOFError,
        SyntaxError,
    ) as error:
        raise ValueError("Could not read the tabletop PNG") from error
    colors, counts = np.unique(first.reshape(-1, 3), axis=0, return_counts=True)
    background = colors[counts.argmax()]
    border = first[0, 0]
    if (
        len(colors) > 64
        or counts.max() < 2048
        or not all(
            np.all(edge == border) for edge in (first[0], first[-1], first[:, 0], first[:, -1])
        )
    ):
        raise ValueError("Use two separated sprites on a flat tabletop with a uniform outer border")
    foreground = ~np.all(first == background, axis=-1) & ~np.all(first == border, axis=-1)
    parts = _components(foreground)
    if len(parts) != 2 or not np.array_equal(np.logical_or.reduce(parts), foreground):
        raise ValueError("This checkpoint requires exactly two separated visible objects")
    observation = observe(first, parts[0])
    # Both selections must be valid for the unchanged offline sampler.
    for mask in observation.masks:
        observe(first, mask)
    return {
        "frames": first[None],
        "frame_count": 32,
        "instance_masks": observation.masks[None],
        "object_valid": np.ones(2, dtype=bool),
        "selected_object_id": 0,
        "force_xy": np.array([0.15, 0], dtype=np.float32),
        "input_source": "uploaded pixels; split and physical properties unknown",
    }
