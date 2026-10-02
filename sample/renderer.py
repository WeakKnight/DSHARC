"""Nan-style explicit compute passes: path trace -> accumulate -> tone map."""
from __future__ import annotations

from dataclasses import dataclass, astuple, replace
import math
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
import slangpy as spy

from camera import Camera
from scene import Scene
from sharc import SharcCache, SharcSettings
from dsharc import DSharcCache, DSharcSettings

SHADER_DIR = Path(__file__).resolve().parent / "shaders"


@dataclass
class Lighting:
    sun_azimuth: float = -65.0
    sun_elevation: float = 35.0
    sun_intensity: float = 3.0
    sun_radius: float = .26785
    sky_intensity: float = .7
    sun_color: tuple = (1.0, .93, .80)
    sky_zenith: tuple = (.12, .30, .65)
    sky_horizon: tuple = (.68, .78, .94)
    sky_ground: tuple = (.035, .028, .022)

    def shader_values(self):
        if not all(math.isfinite(x) for x in (
            self.sun_azimuth, self.sun_elevation, self.sun_intensity, self.sun_radius, self.sky_intensity)):
            raise ValueError("Lighting settings must be finite")
        if not .05 <= self.sun_radius <= 30:
            raise ValueError("Sun radius must be in [0.05, 30] degrees")
        if self.sun_intensity < 0 or self.sky_intensity < 0:
            raise ValueError("Lighting intensities must be nonnegative")
        azimuth, elevation = math.radians(self.sun_azimuth), math.radians(self.sun_elevation)
        return {
            "sun_direction": spy.float3(math.sin(azimuth) * math.cos(elevation),
                math.sin(elevation), math.cos(azimuth) * math.cos(elevation)),
            "sun_color": spy.float3(*self.sun_color), "sun_intensity": self.sun_intensity,
            "sun_cos_radius": math.cos(math.radians(self.sun_radius)),
            "sky_intensity": self.sky_intensity, "sky_zenith": spy.float3(*self.sky_zenith),
            "sky_horizon": spy.float3(*self.sky_horizon), "sky_ground": spy.float3(*self.sky_ground),
        }


def create_device(backend="d3d12", debug=False):
    compiler_options = {"include_paths": [SHADER_DIR],
        "defines": {"DSHARC_SPLIT_KEY_ATOMICS": "1" if backend == "metal" else "0"}}
    if backend == "metal":
        compiler_options["capabilities"] = ["metallib_4_0"]
    return spy.Device(type=getattr(spy.DeviceType, backend), enable_debug_layers=debug,
        compiler_options=compiler_options)


class PathTracer:
    def __init__(self, device: spy.Device, scene: Scene, camera: Camera, lighting: Lighting,
                 width=960, height=640, max_bounces=8, samples_per_frame=1, seed=1,
                 mode="reference", sharc_settings=None, dsharc_settings=None):
        self.device, self.scene, self.camera, self.lighting = device, scene, camera, lighting
        self.max_bounces, self.samples_per_frame, self.seed = max_bounces, samples_per_frame, seed
        self.exposure = 0.0
        self.sample_count = 0
        self._sample_offset = 0
        self._signature = None
        self.mode = mode
        self.sharc_settings = sharc_settings or SharcSettings()
        self.cache = None
        self.dsharc_settings = dsharc_settings or DSharcSettings()
        self.demand_cache = None
        self.path_program = device.load_program("path_tracer.slang", ["compute_main"])
        self.path_pipeline = device.create_compute_pipeline(self.path_program)
        self.accumulate = device.create_compute_kernel(device.load_program("accumulator.slang", ["compute_main"]))
        self.tonemap = device.create_compute_kernel(device.load_program("tone_mapper.slang", ["compute_main"]))
        self.width, self.height = 0, 0
        self.resize(width, height)

    def resize(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Render dimensions must be positive")
        if (width, height) == (self.width, self.height):
            return
        self.device.wait()
        self.width, self.height = width, height
        for name in ("sample", "history", "display", "cache_stats", "update_stats"):
            setattr(self, name, self.device.create_texture(format=spy.Format.rgba32_float,
                width=width, height=height,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label=f"sample.{name}"))
        self.reset()

    def reset(self):
        self.sample_count = 0
        self._sample_offset = 0
        self._signature = None

    def reset_image(self):
        """Restart the displayed mean, retaining cache and a fresh RNG sequence."""
        self._sample_offset += self.sample_count
        self.sample_count = 0

    def bind_path(self, cursor, lighting_values):
        self.scene.bind(cursor.g_scene)
        cursor.g_camera = self.camera.shader_values()
        cursor.g_lighting = lighting_values
        cursor.g_sample_base = (self._sample_offset + self.sample_count) % (2**32)
        cursor.g_samples_per_frame = self.samples_per_frame
        cursor.g_max_bounces = self.max_bounces
        cursor.g_seed = self.seed
        cursor.g_output = self.sample

    def cache_statistics(self):
        if self.mode == "dsharc" and self.demand_cache is not None:
            return self.demand_cache.statistics()
        if self.mode != "sharc" or self.cache is None:
            return {}
        self.device.wait()
        stats = self.cache_stats.to_numpy()[..., :3].mean(axis=(0, 1))
        n = self.sharc_settings.downscale
        update = self.update_stats.to_numpy()[:(self.height+n-1)//n, :(self.width+n-1)//n, :3].mean(axis=(0, 1))
        return {"terminated_path_fraction": float(stats[0]),
                "resampled_update_fraction": float(update[0]),
                "failed_insertions_per_update_path": float(update[1]),
                "segments_per_update_path": float(update[2]),
                "eligible_query_hit_rate": float(stats[0] / max(stats[1], 1e-8)),
                "segments_per_query_path": float(stats[2]),
                "occupied_entries": self.cache.occupancy(),
                "capacity": self.sharc_settings.capacity}

    def render(self, encoder):
        if not 1 <= self.max_bounces <= 32 or not 1 <= self.samples_per_frame <= 64:
            raise ValueError("Expected 1..32 bounces and 1..64 samples per frame")
        lighting_values = self.lighting.shader_values()
        if self.mode not in ("reference", "sharc", "dsharc"):
            raise ValueError("Renderer mode must be reference, sharc or dsharc")
        self.sharc_settings.validate()
        self.dsharc_settings.validate()
        if self.mode == "dsharc" and (self.demand_cache is None or
                self.demand_cache.settings.capacity != self.dsharc_settings.capacity):
            self.device.wait()
            self.demand_cache = DSharcCache(self.device, replace(self.dsharc_settings))
            self._signature = None
        if self.mode == "sharc" and (self.cache is None or
                self.cache.settings.capacity != self.sharc_settings.capacity):
            self.device.wait()
            try:
                self.cache = SharcCache(self.device, replace(self.sharc_settings))
            except RuntimeError as error:
                warnings.warn(f"SHARC initialization failed; using reference PT: {error}", RuntimeWarning)
                self.cache = None
                self.mode = "reference"
            self._signature = None
        signature = (id(self.scene), self.camera.signature(), astuple(self.lighting), self.max_bounces,
                     self.seed, self.mode, astuple(self.sharc_settings), astuple(self.dsharc_settings))
        if signature != self._signature or self.sample_count + self.samples_per_frame >= 2**24:
            self.sample_count = 0
            self._sample_offset = 0
            self._signature = signature
            if self.mode == "sharc":
                self.cache.settings = replace(self.sharc_settings)
                self.cache.clear(encoder)
            if self.mode == "dsharc":
                self.demand_cache.settings = replace(self.dsharc_settings)
                self.demand_cache.clear(encoder)
        if self.mode == "sharc":
            self.cache.render(encoder, self, lighting_values)
        elif self.mode == "dsharc":
            self.demand_cache.render(encoder, self, lighting_values)
        else:
            with encoder.begin_compute_pass() as compute:
                cursor = spy.ShaderCursor(compute.bind_pipeline(self.path_pipeline))
                self.bind_path(cursor, lighting_values)
                compute.dispatch(thread_count=[self.width, self.height, 1])
        # SlangPy tracks texture states/dependencies between these compute passes.
        self.accumulate.dispatch(thread_count=[self.width, self.height, 1],
            vars={"g_sample": self.sample, "g_history": self.history,
                  "g_previous_samples": self.sample_count, "g_new_samples": self.samples_per_frame},
            command_encoder=encoder)
        self.tonemap.dispatch(thread_count=[self.width, self.height, 1],
            vars={"g_input": self.history, "g_output": self.display, "g_exposure_ev": self.exposure},
            command_encoder=encoder)
        self.sample_count += self.samples_per_frame

    def frame(self):
        encoder = self.device.create_command_encoder()
        self.render(encoder)
        self.device.submit_command_buffer(encoder.finish())

    def linear_image(self):
        self.device.wait()
        return np.array(self.history.to_numpy()[..., :3], copy=True)

    def save(self, path: Path, linear_path: Path | None = None):
        self.device.wait()
        display = self.display.to_numpy()[..., :3]
        if not np.isfinite(display).all():
            raise RuntimeError("Renderer produced nonfinite display values")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The display shader already applies sRGB encoding; do not gamma it twice.
        Image.fromarray(np.uint8(np.clip(display * 255.0 + .5, 0, 255))).save(path)
        if linear_path is not None:
            linear_path = Path(linear_path)
            linear_path.parent.mkdir(parents=True, exist_ok=True)
            with linear_path.open("wb") as stream:
                np.save(stream, self.linear_image())
