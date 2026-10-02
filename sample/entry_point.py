"""Run from any directory: python sample/entry_point.py [--headless]."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from camera import DEFAULT_FOV


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SlangPy path tracer with skylight and a finite solar disk")
    parser.add_argument("--headless", action="store_true", help="Render without a window")
    parser.add_argument("--scene", type=Path, help="Optional glTF/GLB, OBJ or other trimesh scene; default: built-in room")
    parser.add_argument("--scene-scale", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--frames", type=int, default=128, help="Accumulated frames in headless mode")
    parser.add_argument("--spp", type=int, default=1, help="Samples per pixel per frame, 1..64")
    parser.add_argument("--max-bounces", type=int, default=32, help="Maximum scattering events, 1..32; use 32 for reference comparisons")
    parser.add_argument("--renderer", choices=("sharc", "reference", "dsharc"), default="sharc")
    parser.add_argument("--dsharc-capacity", type=int, default=1 << 20)
    parser.add_argument("--dsharc-cell-size", type=float, default=.025)
    parser.add_argument("--dsharc-history", type=int, default=256)
    parser.add_argument("--sharc-capacity", type=int, default=1 << 22)
    parser.add_argument("--sharc-downscale", type=int, default=5)
    parser.add_argument("--sharc-grid-scale", type=float, default=80.0, help="Larger means smaller world-space cells")
    parser.add_argument("--sharc-history", type=int, default=256)
    parser.add_argument("--sharc-radiance-scale", type=float, default=65536.0,
                        help="SDK integer accumulation scale; lower for very bright scenes")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output" / "render.png")
    parser.add_argument("--linear-output", type=Path, help="Also save unexposed float32 RGB as a .npy file")
    parser.add_argument("--sun-azimuth", type=float, default=-65.0)
    parser.add_argument("--sun-elevation", type=float, default=35.0)
    parser.add_argument("--sun-intensity", type=float, default=3.0, help="Perpendicular solar irradiance")
    parser.add_argument("--sun-radius", type=float, default=.26785, help="Solar angular radius in degrees, .05..30")
    parser.add_argument("--sky-intensity", type=float, default=.7)
    parser.add_argument("--exposure", type=float, default=1.0, help="Display exposure in EV")
    parser.add_argument("--fov", type=float, default=DEFAULT_FOV, help="Vertical field of view in degrees")
    parser.add_argument("--camera-position", type=float, nargs=3)
    parser.add_argument("--camera-target", type=float, nargs=3)
    parser.add_argument("--backend", choices=("d3d12", "vulkan", "metal"),
                        default="d3d12" if sys.platform == "win32" else "metal" if sys.platform == "darwin" else "vulkan")
    parser.add_argument("--debug", action="store_true", help="Enable graphics API validation")
    parser.add_argument("--vsync", action="store_true")
    parser.add_argument("--window-frames", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from sharc import SharcSettings
    from dsharc import DSharcSettings
    try:
        SharcSettings(args.sharc_capacity, args.sharc_downscale,
                      args.sharc_grid_scale, args.sharc_history, args.sharc_radiance_scale).validate()
        DSharcSettings(capacity=args.dsharc_capacity, cell_size=args.dsharc_cell_size,
                       history=args.dsharc_history).validate()
    except ValueError as error:
        parser.error(str(error))
    for name in ("scene_scale", "sun_azimuth", "sun_elevation", "sun_intensity", "sun_radius", "sky_intensity", "exposure", "fov"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    if not (1 <= args.width <= 8192 and 1 <= args.height <= 8192):
        parser.error("Width/height must be in [1, 8192]")
    if args.frames < 1 or not 1 <= args.spp <= 64 or not 1 <= args.max_bounces <= 32:
        parser.error("Expected positive frames, 1..64 spp and 1..32 bounces")
    if args.scene_scale <= 0 or not .05 <= args.sun_radius <= 30 or not 1 <= args.fov < 179:
        parser.error("Expected positive scale, .05..30 degree sun radius and 1..179 degree FOV")
    if min(args.sun_intensity, args.sky_intensity) < 0 or not -90 <= args.sun_elevation <= 90:
        parser.error("Intensities must be nonnegative and elevation in [-90, 90]")
    if not -20 <= args.exposure <= 20 or not 0 <= args.seed < 2**32 or args.window_frames < 0:
        parser.error("Expected exposure in [-20,20], uint32 seed and nonnegative window frames")
    if bool(args.camera_position) != bool(args.camera_target):
        parser.error("Supply both --camera-position and --camera-target")
    if args.camera_position and not all(math.isfinite(x) for x in args.camera_position + args.camera_target):
        parser.error("Camera coordinates must be finite")
    return args


def main():
    args = parse_args()
    from app import App
    App(args).run()


if __name__ == "__main__":
    main()
