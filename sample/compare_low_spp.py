"""Cold/warm SHARC vs reference PT at 1 spp/frame against a high-spp HDR target."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from camera import Camera
from renderer import Lighting, PathTracer, create_device
from scene import Scene, demo_scene
from compare_sharc import metrics


def display(rgb):
    x = np.maximum(rgb * 2, 0)  # +1 EV, same curve as tone_mapper.slang
    x = np.clip((x * (2.51*x+.03)) / (x*(2.43*x+.59)+.14), 0, 1)
    x = np.where(x <= .0031308, x*12.92, 1.055*x**(1/2.4)-.055)
    return Image.fromarray(np.uint8(np.clip(x*255+.5, 0, 255)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="d3d12", choices=("d3d12", "vulkan"))
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--warmup", type=int, default=256, help="Extra 1-spp SHARC frames; excluded from warm image")
    p.add_argument("--dsharc", action="store_true", help="Also measure warmed DSHARC")
    p.add_argument("--reference", type=Path, help="Existing matching default-room 32768-spp HDR .npy")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "sharc_32spp")
    args = p.parse_args()
    if min(args.width, args.height) < 8 or args.warmup < 0:
        p.error("Dimensions must be >=8 and warmup nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    d = create_device(args.backend)
    scene = Scene(d, demo_scene())

    def tracer(mode, spp=1, seed=71):
        return PathTracer(d, scene, Camera(), Lighting(), args.width, args.height,
            max_bounces=32, samples_per_frame=spp, seed=seed, mode=mode)

    if args.reference:
        target = np.load(args.reference)
    else:
        r = tracer("reference", 32, 1)
        for i in range(1024):
            r.frame()
            if i % 32 == 31:
                d.wait()
        target = r.linear_image()
        np.save(args.output / "reference_32768.npy", target)
        del r
    if target.shape != (args.height, args.width, 3) or not np.isfinite(target).all():
        raise ValueError("Reference must match dimensions and contain finite linear HDR")
    display(target).save(args.output / "reference_32768.png")
    results = {"settings": vars(args) | {"output": str(args.output), "reference": str(args.reference),
        "spp_per_frame": 1, "max_bounces": 32, "resampling": True,
        "update_paths_per_frame": ((args.width+4)//5)*((args.height+4)//5)}, "adapter": d.info.adapter_name}
    from sharc import SharcSettings
    from dsharc import DSharcSettings
    results["settings"]["sharc_settings"] = asdict(SharcSettings())
    results["settings"]["dsharc_settings"] = asdict(DSharcSettings())
    modes = [("off", "reference"), ("on_cold", "sharc"), ("on_warm", "sharc")]
    if args.dsharc:
        modes.append(("dsharc_warm", "dsharc"))
    for name, mode in modes:
        r = tracer(mode)
        r.exposure = 1
        if name in ("on_warm", "dsharc_warm"):
            for i in range(args.warmup):
                r.frame()
                if i % 32 == 31:
                    d.wait()
            r.reset_image()  # Does not re-use the warmup's image samples or RNG sequence.
        checkpoints = {}
        for frame in range(32):
            r.frame()
            if frame+1 in (1, 4, 8, 16, 32):
                value = r.linear_image()
                checkpoints[str(frame+1)] = metrics(value, target) | {"cache": r.cache_statistics()}
                r.save(args.output / f"{name}_{frame+1}.png", args.output / f"{name}_{frame+1}.npy")
        results[name] = checkpoints
        print(name, json.dumps(checkpoints["32"]), flush=True)
        del r
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # Four full-size views, identical exposure; no image filtering or denoising.
    w, h = args.width, args.height
    canvas = Image.new("RGB", (w*2, (h+32)*2), "#202020")
    draw = ImageDraw.Draw(canvas)
    rows = [("off_32.png", "SHARC OFF | 32 spp"),
            ("on_cold_32.png", "SHARC ON | cold cache | 32 spp"),
            ("on_warm_32.png", f"SHARC ON | {args.warmup} warmup frames + 32 spp"),
            ("reference_32768.png", "Reference PT | 32768 spp")]
    for i, (filename, label) in enumerate(rows):
        x, y = (i % 2)*w, (i // 2)*(h+32)
        draw.text((x+8, y+9), label, fill="white")
        canvas.paste(Image.open(args.output/filename), (x, y+32))
    canvas.save(args.output / "comparison_32spp.png")
    if args.dsharc:
        for spp in (1, 32):
            rows = [(f"off_{spp}.png", f"Reference PT | {spp} spp"),
                    (f"on_warm_{spp}.png", f"SHARC | {args.warmup} warmup + {spp} spp"),
                    (f"dsharc_warm_{spp}.png", f"DSHARC | {args.warmup} warmup + {spp} spp"),
                    ("reference_32768.png", "Reference PT | 32768 spp")]
            for i, (filename, label) in enumerate(rows):
                x, y = (i % 2)*w, (i // 2)*(h+32)
                draw.rectangle((x,y,x+w,y+32), fill="#202020")
                draw.text((x+8,y+9), label, fill="white")
                canvas.paste(Image.open(args.output/filename),(x,y+32))
            canvas.save(args.output / f"comparison_{spp}spp_dsharc.png")


if __name__ == "__main__":
    main()
