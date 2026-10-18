"""Optional real-browser rehearsal against a running local checkpoint server.

Run with: uv run --no-project --with playwright python scripts/check_demo_browser.py
Uses installed Chrome/Edge in a fresh headless session; downloads no browser.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo-browser"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-live", action="store_true")
    parser.add_argument("--browser-executable", type=Path)
    args = parser.parse_args()
    if urlsplit(args.url).hostname not in {"localhost", "127.0.0.1"}:
        parser.error("Use a local demo server")
    if not 0 <= args.seed <= 2147473644:
        parser.error("Seed must leave room for the cancellation probe")
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
        parser.error("This rehearsal needs installed Chrome or Edge")
    args.output.mkdir(parents=True, exist_ok=True)
    checks, errors, latency, sources = [], [], [], []
    with sync_playwright() as runtime:
        browser = runtime.chromium.launch(executable_path=str(executable), headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1080}, device_scale_factor=1)
        page.on("pageerror", lambda error: errors.append(str(error)))

        def wait(expression, timeout=30000):
            deadline = time.monotonic() + timeout / 1000
            while time.monotonic() < deadline:
                if page.evaluate(expression):
                    return
                time.sleep(0.1)
            raise AssertionError("Browser condition timed out: " + expression)

        page.goto(args.url)
        page.locator("#generate").wait_for(state="visible")
        wait("state && scene && !busy")
        assert page.title() == "click2future"
        checks.append("page and checkpoint metadata load")
        assert page.locator("#model").input_value() == page.evaluate("state.models[0].id")
        checks.append("one primary model is selected by default")
        assert page.locator("#objects button").count() >= 2
        checks.append("visible object selectors")
        page.screenshot(path=str(args.output / "01-ready.png"), full_page=True)

        canvas = page.locator("#input")
        original = page.evaluate("({selected,force:force.slice(),scene:scene.id})")
        canvas.click(position={"x": 12, "y": 12})
        assert page.evaluate("selected") == original["selected"]
        checks.append("background click cannot select an object")
        page.locator("#objects button").first.click()
        assert page.locator("#objects button").first.get_attribute("aria-pressed") == "true"
        checks.append("object buttons update selection")

        def drag_to(dx, dy):
            center = page.evaluate("scene.objects.find(obj=>obj.id===selected).center")
            size = page.evaluate("scene.size")
            box = canvas.bounding_box()
            x, y = (
                box["x"] + center[0] * box["width"] / size[1],
                box["y"] + center[1] * box["height"] / size[0],
            )
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + dx, y + dy, steps=5)
            page.mouse.up()
            return page.evaluate("force")

        force = drag_to(60, 0)
        assert force[0] > 0 and abs(force[1]) < 1e-6
        checks.append("right drag gives positive x")
        force = drag_to(0, 60)
        assert force[1] > 0 and abs(force[0]) < 1e-6
        checks.append("down drag gives positive y")
        force = drag_to(600, 600)
        assert sum(value * value for value in force) <= 0.250001
        checks.append("drag outside canvas clamps physical force")
        page.locator("#reverse").click()
        reversed_force = page.evaluate("force")
        assert all(abs(a + b) < 1e-6 for a, b in zip(force, reversed_force, strict=True))
        checks.append("reverse keeps magnitude and negates both coordinates")
        page.locator("#zero").click()
        assert page.evaluate("force") == [0, 0]
        checks.append("zero push")
        canvas.focus()
        page.keyboard.press("ArrowUp")
        assert abs(page.evaluate("force[1]") + 0.025) < 1e-6
        checks.append("keyboard direction matches image coordinates")
        page.keyboard.press("Shift+ArrowDown")
        assert abs(page.evaluate("force[1]") + 0.020) < 1e-6
        checks.append("shift fine adjustment")
        page.locator("#seed").fill("-1")
        page.locator("#generate").click()
        assert "whole-number seed" in page.locator("#status").inner_text()
        assert not page.evaluate("busy")
        checks.append("invalid seed fails without model execution")

        # Reload restores the exact original condition rather than constructing one
        # in the browser test. Every generation goes through the real UI request.
        page.reload()
        wait("state && scene && !busy")
        page.locator("#seed").fill(str(args.seed))

        def generate():
            started = time.perf_counter()
            page.locator("#generate").click()
            wait("busy || playback")
            assert page.locator("#scene").is_disabled()
            assert page.locator("#model").is_disabled()
            assert page.locator("#seed").is_disabled()
            wait("!busy", timeout=180000)
            assert "ready." in page.locator("#status").inner_text(), page.locator(
                "#status"
            ).inner_text()
            latency.append(time.perf_counter() - started)
            sources.append(
                page.evaluate(
                    "playback.models.map(model=>({source:model.source,seconds:model.seconds}))"
                )
            )

        generate()
        if args.require_live:
            assert sources[0][0]["source"] == "live model", (
                "use a seed absent from the replay cache"
            )
        checks.append("real checkpoint finishes four seeded futures")
        checks.append("controls locked during generation")
        assert len(page.evaluate("playback.models")) == 1
        assert page.evaluate("playing")
        wait("frame > 0")
        checks.append("native-fps playback advances without an extra click")
        assert page.locator(".future-card canvas").count() == 4 * len(
            page.evaluate("playback.models")
        )
        assert page.evaluate("playback.seeds") == list(range(args.seed, args.seed + 4))
        assert page.evaluate("playback.frames") == 32
        checks.append("four videos retain ordered paired seeds and 32 frames")
        page.locator("#timeline").focus()
        page.keyboard.press("End")
        assert page.locator("#time").inner_text() == "3.88s"
        assert not page.evaluate("playing")
        checks.append("shared timeline scrubs all samples and pauses")
        page.locator("#tracks").check()
        assert page.locator(".track-count").count() == page.locator(".future-card").count()
        assert page.evaluate("playback.models.every(model=>model.pixel_tracks)")
        checks.append("pixel-track overlay and missing-count labels")
        page.locator("#tracks").uncheck()
        page.locator("#timeline").focus()
        page.keyboard.press("Home")
        assert page.locator("#time").inner_text() == "0.00s"
        checks.append("timeline returns to supplied frame zero")
        page.locator("#reference-details summary").click()
        assert page.locator("#reference").is_visible()
        assert "simulator reference" in page.evaluate("playback.reference.source")
        checks.append("simulator reference is separate and labelled")
        page.locator("#reference-details summary").click()
        for width, height in ((1440, 1080), (1440, 900), (1366, 768)):
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("window.scrollTo(0,0)")
            bottoms = page.locator(".future-card").evaluate_all(
                "cards => cards.map(card => card.getBoundingClientRect().bottom)"
            )
            assert max(bottoms) <= height, (width, height, bottoms)
        checks.append("all four primary futures fit common desktop viewport heights")
        page.set_viewport_size({"width": 1440, "height": 1080})
        page.screenshot(path=str(args.output / "02-futures.png"), full_page=True)
        first = page.evaluate("playback.models[0].strip")
        generate()
        assert page.locator("#mode").inner_text() == "cached replay"
        assert page.evaluate("playback.models[0].strip") == first
        checks.append("exact repeated request is identical labelled replay")
        page.emulate_media(reduced_motion="reduce")
        generate()
        assert not page.evaluate("playing")
        page.emulate_media(reduced_motion="no-preference")
        checks.append("reduced-motion preference keeps playback paused")
        if len(page.evaluate("state.models")) > 1:
            page.locator("#model").select_option("__all")
            generate()
            assert page.locator(".future-card canvas").count() == 4 * len(
                page.evaluate("state.models")
            )
            checks.append("explicit comparison includes every selected model")
        page.locator("#reverse").click()
        assert "previous request" in page.locator("#mode").inner_text()
        assert page.evaluate("playback.models[0].strip") == first
        checks.append("changed push cannot relabel old videos as new output")
        page.locator("#seed").fill(str(args.seed + 10000))
        page.locator("#generate").click()
        wait("busy")
        page.locator("#cancel").click()
        wait("!busy")
        page.wait_for_timeout(1000)
        assert page.evaluate("playback.models[0].strip") == first
        assert "cancelled" in page.locator("#status").inner_text()
        checks.append("cancelled generation cannot replace the previous output")
        options = page.locator("#scene option")
        if options.count() > 1:
            page.locator("#scene").select_option(index=1)
            wait("!busy && !playback")
            assert page.locator(".future-card").count() == 0
            assert page.locator("#timeline").is_disabled()
            checks.append("scene change resets playback and output")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(args.output / "03-mobile.png"), full_page=True)
        checks.append("mobile layout has no horizontal overflow")
        assert not errors, errors
        checks.append("no browser javascript exceptions")
        record = {
            "checks": checks,
            "count": len(checks),
            "browser_errors": errors,
            "ui_seconds": latency,
            "model_sources": sources,
            "note": "functional rehearsal, not model quality or a latency benchmark",
        }
        (args.output / "checks.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record))
        browser.close()


if __name__ == "__main__":
    main()
