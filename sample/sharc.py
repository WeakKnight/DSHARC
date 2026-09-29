"""NVIDIA SHARC 1.8.3 resources and explicit Update -> Resolve -> Query passes."""
from dataclasses import dataclass
import math
import numpy as np
import slangpy as spy


@dataclass
class SharcSettings:
    capacity: int = 1 << 22
    downscale: int = 5
    scene_scale: float = 80.0
    history: int = 256
    radiance_scale: float = 65536.0

    def validate(self):
        if self.capacity < 1024 or self.capacity > 1 << 24 or self.capacity & (self.capacity - 1):
            raise ValueError("SHARC capacity must be a power of two in [1024, 2^24]")
        if not 1 <= self.downscale <= 10 or not 1 <= self.history <= 1024:
            raise ValueError("SHARC downscale must be 1..10 and history 1..1024")
        if not all(math.isfinite(v) and v > 0 for v in (self.scene_scale, self.radiance_scale)):
            raise ValueError("SHARC grid/radiance scales must be finite and positive")


class SharcCache:
    def __init__(self, device, settings):
        settings.validate()
        self.device, self.settings = device, settings
        self.programs = {name: device.load_program(f"sharc_{name}.slang", ["compute_main"])
                         for name in ("update", "resolve", "query")}
        self.pipelines = {name: device.create_compute_pipeline(p) for name, p in self.programs.items()}
        # Native Slang float16_t4 is 8 bytes on DXIL and SPIR-V. Allocate from
        # reflected resource layouts, rather than assuming Python struct packing.
        layout = spy.ReflectionCursor(self.programs["update"])
        self.buffers = {}
        for field, expected in (("hashGridData.hashEntriesBuffer", 8),
                                ("accumulationBuffer", 16), ("resolvedBuffer", 16)):
            reflection = layout["g_sharc"]
            for part in field.split("."):
                reflection = reflection[part]
            buffer = device.create_buffer(resource_type_layout=reflection.type_layout, element_count=settings.capacity,
                usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
                label=f"SHARC.{field}")
            if buffer.size != settings.capacity * expected:
                raise RuntimeError(f"Unexpected SHARC layout for {field}: {buffer.size}")
            self.buffers[field] = buffer
        self.frame_index = 0

    def clear(self, encoder):
        for buffer in self.buffers.values():
            encoder.clear_buffer(buffer)
        encoder.global_barrier()
        self.frame_index = 0

    def bind(self, cursor, renderer):
        p = cursor.g_sharc
        p.hashGridParameters = {"cameraPosition": renderer.camera.shader_values()["position"],
            "logarithmBase": 2.0, "sceneScale": self.settings.scene_scale, "levelBias": 0.0}
        p.hashGridData.capacity = self.settings.capacity
        p.hashGridData.hashEntriesBuffer = self.buffers["hashGridData.hashEntriesBuffer"]
        p.accumulationBuffer = self.buffers["accumulationBuffer"]
        p.resolvedBuffer = self.buffers["resolvedBuffer"]
        p.radianceScale = self.settings.radiance_scale
        cursor.g_cache_frame = self.frame_index
        cursor.g_cache_history = self.settings.history
        cursor.g_update_downscale = self.settings.downscale
        cursor.g_cache_stats = renderer.cache_stats
        cursor.g_update_stats = renderer.update_stats

    def render(self, encoder, renderer, lighting):
        n = self.settings.downscale
        for name in ("update", "resolve", "query"):
            with encoder.begin_compute_pass() as compute:
                cursor = spy.ShaderCursor(compute.bind_pipeline(self.pipelines[name]))
                self.bind(cursor, renderer)
                if name == "resolve":
                    size = [self.settings.capacity, 1, 1]
                else:
                    renderer.bind_path(cursor, lighting)
                    size = [(renderer.width + n - 1) // n, (renderer.height + n - 1) // n, 1] \
                        if name == "update" else [renderer.width, renderer.height, 1]
                compute.dispatch(thread_count=size)
            # Explicit UAV visibility: Update writes -> Resolve reads/writes;
            # Resolve writes -> Query reads. Also orders reuse on the next frame.
            encoder.global_barrier()
        self.frame_index += 1

    def occupancy(self):
        self.device.wait()
        keys = self.buffers["hashGridData.hashEntriesBuffer"].to_numpy().view(np.uint64)
        return int(np.count_nonzero(keys))
