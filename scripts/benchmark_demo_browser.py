"""Warm end-to-end browser timing against a running local demo (fresh seeds only)."""

import argparse
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = min(len(ordered) - 1, lower + 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=61017000)
    parser.add_argument("--browser-executable", type=Path)
    args = parser.parse_args()
    if (
        urlsplit(args.url).hostname not in {"localhost", "127.0.0.1"}
        or not 3 <= args.repeats <= 20
        or not 0 <= args.seed <= 2147483500
    ):
        parser.error("Use a local server, 3-20 measurements and a valid seed range")
    if args.output.exists():
        parser.error("Keep previous timing records; use a fresh output path")
    candidates = (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files"))
        / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe",
        *(
            Path(path)
            for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge")
            if (path := shutil.which(name))
        ),
    )
    executable = args.browser_executable or next(
        (path for path in candidates if path.is_file()), None
    )
    if executable is None or not executable.is_file():
        parser.error("This benchmark needs installed Chrome or Edge")
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=str(executable), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        deadline = time.monotonic() + 30
        while not page.evaluate("typeof state !== 'undefined' && state && scene && !busy"):
            if time.monotonic() > deadline:
                raise RuntimeError("Local demo did not initialize")
            time.sleep(0.1)
        metadata = page.evaluate("state.models")
        condition = page.evaluate("({scene:scene.id,object:selected,force:force.slice()})")
        runs = []
        for index in range(args.repeats + 1):
            seed = args.seed + index * 4
            page.locator("#seed").fill(str(seed))
            started = time.perf_counter()
            page.locator("#generate").click()
            deadline = time.monotonic() + 180
            while page.evaluate("busy"):
                if time.monotonic() > deadline:
                    raise RuntimeError("Generation timed out")
                time.sleep(0.05)
            if "ready." not in page.locator("#status").inner_text():
                raise RuntimeError(page.locator("#status").inner_text())
            page.evaluate("() => new Promise(resolve=>requestAnimationFrame(()=>resolve(true)))")
            seconds = time.perf_counter() - started
            details = page.evaluate(
                "playback.models.map(model=>({name:model.name,source:model.source,seconds:model.seconds}))"
            )
            if any(model["source"] != "live model" for model in details):
                raise ValueError("Cache hit in live benchmark; choose a fresh seed range")
            runs.append(
                {
                    "warmup": index == 0,
                    "seconds": seconds,
                    "seeds": list(range(seed, seed + 4)),
                    "models": details,
                }
            )
        measured = [row["seconds"] for row in runs if not row["warmup"]]
        record = {
            "scope": "warm browser request through first painted canvas; headless Chrome/Edge",
            "included": (
                "queue, neural sampling, conversion, pixel tracking, reference, "
                "PNG transfer and browser display"
            ),
            "excluded": "server startup and checkpoint loading",
            "condition": condition,
            "models": metadata,
            "runs": runs,
            "n": len(measured),
            "median_seconds": statistics.median(measured),
            "empirical_p95_seconds": percentile(measured, 0.95),
            "quantile_method": (
                "linear interpolation of measured samples; small-n estimate, not a confidence bound"
            ),
            "browser_errors": errors,
        }
        if errors:
            raise RuntimeError(errors)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(
            json.dumps(
                {
                    "n": len(measured),
                    "median_seconds": record["median_seconds"],
                    "empirical_p95_seconds": record["empirical_p95_seconds"],
                }
            )
        )
        browser.close()


if __name__ == "__main__":
    main()
