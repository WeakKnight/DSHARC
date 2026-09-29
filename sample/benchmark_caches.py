"""Steady-state application batch time, including Python submission and GPU work."""
import argparse
import json
from pathlib import Path
import time
from camera import Camera
from renderer import Lighting, PathTracer, create_device
from scene import Scene, demo_scene


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("d3d12", "vulkan"), default="d3d12")
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "cache_timing.json")
    args = p.parse_args()
    d = create_device(args.backend)
    scene = Scene(d, demo_scene())
    results = {"adapter": d.info.adapter_name, "backend": args.backend,
        "width": 480, "height": 320, "spp_per_frame": 1, "warmup_frames": 256,
        "batch_frames": 128, "batches": 3,
        "timing": "CPU+GPU end-to-end batch throughput; no UI/readback/compile inside timing"}
    for mode in ("reference", "sharc", "dsharc"):
        r = PathTracer(d, scene, Camera(), Lighting(), 480, 320, 32, 1, mode=mode)
        for i in range(256):
            r.frame()
            if i % 32 == 31:
                d.wait()
        d.wait()
        batches = []
        for _ in range(3):
            start = time.perf_counter()
            for i in range(128):
                r.frame()
                if i % 32 == 31:
                    d.wait()
            d.wait()
            batches.append((time.perf_counter()-start)*1000/128)
        results[mode] = {"milliseconds_per_frame_batches": batches,
            "mean_milliseconds_per_frame": sum(batches)/len(batches), "cache":r.cache_statistics()}
        del r
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
