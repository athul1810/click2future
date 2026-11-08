import numpy as np
from PIL import Image, ImageDraw

from click2future.evaluation.rigid_geometry import geometry_diagnostics
from click2future.models.contact_projection import project_motion
from click2future.models.object_motion import observe


def observation(centers):
    image = Image.new("RGB", (64, 64), (210, 215, 220))
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(centers, [(170, 60, 50), (50, 100, 190)], strict=True):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    first = np.asarray(image)
    return observe(first, np.all(first == [170, 60, 50], axis=-1))


def moving(obs, dx_pixels, frames=3):
    motion = np.zeros((frames, 2, 4), dtype=np.float32)
    motion[:, obs.selected_index, 0] = dx_pixels / 32
    return motion


def test_separates_overlap_by_inverse_mass_and_removes_penetration():
    obs = observation([(20, 32), (32, 32)])
    motion = moving(obs, 8.0)
    raw = geometry_diagnostics(obs, motion[None])
    assert raw["clips_with_penetration_over_one_pixel"] == 1
    projected = project_motion(obs, motion, [1.0, 3.0])
    fixed = geometry_diagnostics(obs, projected[None])
    assert fixed["clips_with_penetration_over_one_pixel"] == 0
    shift = (projected - motion)[..., :2]
    light, heavy = np.abs(shift[:, 0, 0]).max(), np.abs(shift[:, 1, 0]).max()
    assert np.isclose(light, 3 * heavy, rtol=1e-3)


def test_frames_without_violation_and_rest_are_unchanged():
    obs = observation([(16, 16), (48, 48)])
    for motion in (np.zeros((4, 2, 4), dtype=np.float32), moving(obs, 3.0, frames=4)):
        assert np.array_equal(project_motion(obs, motion, [1.0, 1.0]), motion)


def test_keeps_hulls_on_the_canvas():
    obs = observation([(8, 32), (48, 32)])
    motion = moving(obs, -10.0)
    projected = project_motion(obs, motion, [1.0, 1.0])
    report = geometry_diagnostics(obs, projected[None])
    assert report["clips"][0]["boundary_over_one_pixel_object_frames"] == 0
