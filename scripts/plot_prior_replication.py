"""Show every declared seed for the first sorted D018 condition, without selection."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True, help="Completed D018 result directory")
parser.add_argument("--output", type=Path, required=True, help="New PNG path")
args = parser.parse_args()
RUN = args.root
if args.output.exists():
    parser.error("Keep previous figures and use a new output file")
plan = json.loads((RUN / "plan.json").read_text())
case = plan["cases"][0]
assert len(case["reference_ids"]) == 4
panels = []
references = []
for name in case["reference_ids"]:
    with np.load(Path(plan["dataset"]) / (name + ".npz"), allow_pickle=False) as arrays:
        references.append(arrays["frames"])
panels.append(("simulator reference variants", references, "world"))
for arm, label in [
    ("direct", "one pass, refinement seed 19"),
    ("diffusion", "diffusion, refinement seed 19"),
]:
    with np.load(
        RUN / (arm + "-seed19") / "validation-samples" / (case["sample_id"] + ".npz")
    ) as arrays:
        clips = arrays["videos"].copy()
    assert clips.shape[0] == 4 and np.all(clips[:, 0] == references[0][0])
    panels.append((label, clips, "seed"))


def font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


large = font(32)
small = font(30)
image = Image.new("RGB", (1680, 720), "white")
draw = ImageDraw.Draw(image)
frames = [0, 8, 16, 31]
tile = 112
for panel_index, (title, videos, prefix) in enumerate(panels):
    x0 = panel_index * 560
    draw.text((x0 + 20, 14), title, font=large, fill="#152115")
    for j, frame in enumerate(frames):
        draw.text((x0 + 84 + j * 116, 53), f"{frame / 8:g}s", font=small, fill="#425249")
    for row, video in enumerate(videos):
        y = 92 + row * 128
        draw.text((x0 + 5, y + 40), prefix + "\n" + str(row), font=small, fill="#26332f")
        for j, frame in enumerate(frames):
            x = x0 + 84 + j * 116
            image.paste(
                Image.fromarray(video[frame]).resize((tile, tile), Image.Resampling.NEAREST), (x, y)
            )
            draw.rectangle((x, y, x + tile, y + tile), outline="#778179", width=1)
draw.text(
    (20, 614),
    "reference variants and sampling seeds are separate.\n"
    "row numbers do not identify a shared physical world.",
    font=small,
    fill="#26332f",
)
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output)
(output.with_suffix(".json")).write_text(
    json.dumps(
        {
            "case": case,
            "rule": "first sorted D018 ID condition; no outcome selection",
            "refinement_seed": 19,
            "sampling_seeds": [0, 1, 2, 3],
            "frames": frames,
            "scope": (
                "all four futures per arm plus all four reference variants; "
                "no world correspondence or calibration claim"
            ),
            "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "figure_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
        indent=2,
    )
    + "\n"
)
print(output)
