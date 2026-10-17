"""Local polygon vertices shared by rendering and collision geometry."""

import math


def polygon_vertices(shape, geometry):
    if shape == "triangle":
        radius = float(geometry["radius"])
        return [
            (
                radius * math.cos(-math.pi / 2 + k * 2 * math.pi / 3),
                radius * math.sin(-math.pi / 2 + k * 2 * math.pi / 3),
            )
            for k in range(3)
        ]
    if shape == "rectangle":
        w, h = float(geometry["width"]) / 2, float(geometry["height"]) / 2
        return [(-w, -h), (w, -h), (w, h), (-w, h)]
    raise ValueError(f"unsupported polygon: {shape}")
