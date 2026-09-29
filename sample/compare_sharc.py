"""Reproducible linear-HDR diffuse-room comparison; no denoising/exposure in metrics."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from camera import Camera
from renderer import Lighting, PathTracer, create_device
from scene import Scene, demo_scene
from sharc import SharcSettings
from dsharc import DSharcSettings


def blocks(image, size=8):
    h, w = image.shape[:2]
    return image[:h // size * size, :w // size * size].reshape(
        h // size, size, w // size, size, 3).mean(axis=(1, 3))


def metrics(value, reference):
    luminance = np.array([.2126, .7152, .0722])
    v, r = value @ luminance, reference @ luminance
    vb, rb = blocks(value) @ luminance, blocks(reference) @ luminance
    dark = r < np.median(r)
    local_error = np.abs(vb-rb) / np.maximum(rb, .001)
    return {"mean_luminance": float(v.mean()),
        "mean_bias_percent": float((v.mean() / r.mean() - 1) * 100),
        "rgb_channel_bias_percent": ((value.mean((0, 1)) / reference.mean((0, 1)) - 1) * 100).tolist(),
        "relative_l1_percent": float(np.abs(v-r).mean() / r.mean() * 100),
        "relative_rmse_percent": float(np.sqrt(np.mean((v-r)**2)) / r.mean() * 100),
        "block8_relative_l1_percent": float(np.abs(vb-rb).mean() / rb.mean() * 100),
        "dark_half_mean_bias_percent": float((v[dark].mean() / r[dark].mean() - 1) * 100),
        "block8_local_relative_error_percentiles": dict(zip(("p50", "p90", "p95", "p99", "max"),
            np.percentile(local_error * 100, [50, 90, 95, 99, 100]).tolist())),
        "block8_relative_rmse_percent": float(np.sqrt(np.mean((vb-rb)**2)) / rb.mean() * 100)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="d3d12", choices=("d3d12", "vulkan"))
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--frames", type=int, default=512)
    p.add_argument("--spp", type=int, default=16)
    p.add_argument("--grid-scale", type=float, default=80)
    p.add_argument("--capacity", type=int, default=1 << 22)
    p.add_argument("--cache", choices=("sharc", "dsharc"), default="sharc")
    p.add_argument("--dsharc-capacity", type=int, default=1 << 20)
    p.add_argument("--dsharc-cell-size", type=float, default=.025)
    p.add_argument("--dsharc-history", type=int, default=256)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "sharc_comparison")
    args = p.parse_args()
    if not (8 <= args.width <= 8192 and 8 <= args.height <= 8192 and
            args.frames >= 4 and 1 <= args.spp <= 64):
        p.error("Expected dimensions 8..8192, at least 4 frames, and 1..64 spp")
    args.output.mkdir(parents=True, exist_ok=True)
    device = create_device(args.backend)
    scene = Scene(device, demo_scene())
    results, images = {"adapter": device.info.adapter_name}, {}
    for name, mode, seed in (("reference", "reference", 1),
                             ("reference_independent", "reference", 913),
                             (args.cache, args.cache, 271)):
        renderer = PathTracer(device, scene, Camera(), Lighting(), args.width, args.height,
            32, args.spp, seed, mode=mode,
            sharc_settings=SharcSettings(capacity=args.capacity, scene_scale=args.grid_scale),
            dsharc_settings=DSharcSettings(capacity=args.dsharc_capacity,
                cell_size=args.dsharc_cell_size, history=args.dsharc_history))
        renderer.exposure = 1
        start = time.perf_counter()
        checkpoints = {}
        for frame in range(args.frames):
            renderer.frame()
            if (frame + 1) % 32 == 0:
                device.wait()
            if frame + 1 in (args.frames // 4, args.frames // 2, args.frames):
                image = renderer.linear_image()
                checkpoints[str(renderer.sample_count)] = float((image @ [.2126, .7152, .0722]).mean())
                print(f"{name}: {renderer.sample_count} spp, mean luminance {checkpoints[str(renderer.sample_count)]:.8f}", flush=True)
        device.wait()
        results[name] = {"seconds_including_first_compile": time.perf_counter() - start,
                         "checkpoints": checkpoints, "cache": renderer.cache_statistics()}
        images[name] = renderer.linear_image()
        renderer.save(args.output / f"{name}.png", args.output / f"{name}.npy")
        del renderer
    for name in ("reference_independent", args.cache):
        results[name]["vs_reference"] = metrics(images[name], images["reference"])
    results["settings"] = vars(args) | {"output": str(args.output), "max_bounces": 32}
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    canvas = Image.new("RGB", (args.width * 2, args.height + 28), "#202020")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(("reference", args.cache)):
        canvas.paste(Image.open(args.output / f"{name}.png"), (i * args.width, 28))
        draw.text((i * args.width + 8, 7), f"{name} | {args.frames * args.spp} spp | +1 EV", fill="white")
    canvas.save(args.output / "comparison.png")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
