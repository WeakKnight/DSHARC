"""Renderer adapter for the project's demand-driven DSHARC headers."""
from dataclasses import dataclass
import math
import numpy as np
import slangpy as spy


@dataclass
class DSharcSettings:
    capacity: int = 1 << 20
    cell_size: float = .025
    level_distance: float = 2.0
    history: int = 256
    stale_frames: int = 16

    def validate(self):
        if not 1024 <= self.capacity <= 1 << 21 or self.capacity & (self.capacity - 1):
            raise ValueError("DSHARC capacity must be a power of two in [1024, 2^21]")
        if not all(math.isfinite(x) and x > 0 for x in (self.cell_size, self.level_distance)):
            raise ValueError("DSHARC cell size and level distance must be finite and positive")
        if not 1 <= self.history <= 1024 or not 0 <= self.stale_frames <= 1024:
            raise ValueError("DSHARC history must be 1..1024 and stale frames 0..1024")


class DSharcCache:
    def __init__(self, device, settings):
        settings.validate()
        self.device, self.settings = device, settings
        self.programs = {}
        for name in ("begin", "request", "compact", "args", "update", "resolve", "gather"):
            module, entry = (f"dsharc_{name}.slang", "compute_main") if name in (
                "request", "update", "gather") else ("dsharc_manage.slang", f"{name}_main")
            self.programs[name] = device.load_program(module, [entry])
        self.pipelines = {name: device.create_compute_pipeline(p) for name, p in self.programs.items()}
        self.layout = spy.ReflectionCursor(self.programs["request"])
        self.buffers = {}
        resources = [("keys",8), ("states",16), ("surfaces",32), ("materials",32),
            ("previousRadiance",16), ("currentRadiance",16), ("estimates",16),
            ("activeEntries",4), ("activeCount",4)]
        if device.info.type == spy.DeviceType.metal:
            resources.append(("keyStates",4))
        for name, stride in resources:
            count = 1 if name == "activeCount" else settings.capacity
            b = device.create_buffer(resource_type_layout=self.layout.g_dsharc[name].type_layout,
                element_count=count, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f"DSHARC.{name}")
            if b.size != count * stride:
                raise RuntimeError(f"Unexpected DSHARC stride: {name}")
            self.buffers[name] = b
        self.arguments = device.create_buffer(size=12,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.indirect_argument)
        self.stats = device.create_buffer(size=24, struct_size=4,
            usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource)
        self.paths = None
        self.path_count = 0
        self.frame_index = 0

    def clear(self, encoder):
        for b in self.buffers.values():
            encoder.clear_buffer(b)
        encoder.global_barrier()
        self.frame_index = 0

    def bind(self, cursor, renderer):
        p = cursor.g_dsharc
        p.grid = {"cameraPosition": renderer.camera.shader_values()["position"],
            "origin": spy.float3(0), "baseCellSize": self.settings.cell_size,
            "levelDistance": self.settings.level_distance, "maxLevel": 12}
        p.capacity = self.settings.capacity
        p.frameIndex = self.frame_index
        p.staleFrameCount = self.settings.stale_frames
        p.maxAccumulatedFrames = self.settings.history
        for name, b in self.buffers.items():
            p[name] = b
        cursor.g_paths = self.paths
        cursor.g_dstats = self.stats
        cursor.g_dispatch = self.arguments

    def render(self, encoder, renderer, lighting):
        count = renderer.width * renderer.height * renderer.samples_per_frame
        if count != self.path_count:
            self.device.wait()
            self.paths = self.device.create_buffer(resource_type_layout=self.layout.g_paths.type_layout,
                element_count=count, usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)
            if self.paths.size != count * 32:
                raise RuntimeError("Unexpected DSHARC path record stride")
            self.path_count = count
        encoder.clear_buffer(self.stats)
        encoder.global_barrier()
        for name in ("begin", "request", "compact", "args", "update", "resolve", "gather"):
            with encoder.begin_compute_pass() as compute:
                cursor = spy.ShaderCursor(compute.bind_pipeline(self.pipelines[name]))
                self.bind(cursor, renderer)
                if name in ("request", "update", "gather"):
                    renderer.bind_path(cursor, lighting)
                    if name == "update":
                        cursor.g_max_bounces = 32
                if name in ("update", "resolve"):
                    compute.dispatch_compute_indirect(self.arguments)
                elif name in ("request", "gather"):
                    compute.dispatch(thread_count=[renderer.width, renderer.height, 1])
                else:
                    compute.dispatch(thread_count=[1 if name == "args" else self.settings.capacity, 1, 1])
            # Each phase publishes writes before consumers; argument usage is
            # transitioned by SlangPy. Lighting never reads this frame's result.
            encoder.global_barrier()
        self.buffers["previousRadiance"], self.buffers["currentRadiance"] = (
            self.buffers["currentRadiance"], self.buffers["previousRadiance"])
        self.frame_index = (self.frame_index + 1) % (2**32)

    def statistics(self):
        self.device.wait()
        v = self.stats.to_numpy().view(np.uint32).ravel()
        live = int(self.buffers["activeCount"].to_numpy().view(np.uint32).ravel()[0])
        return {"requested_path_fraction": float(v[0] / max(self.path_count, 1)),
            "failed_requests": int(v[1]), "gather_fallbacks": int(v[2]),
            "feedback_hits": int(v[3]), "feedback_misses": int(v[4]),
            "invalid_materials": int(v[5]), "active_entries": live,
            "capacity": self.settings.capacity, "frame_index": self.frame_index}
