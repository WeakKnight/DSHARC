from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import slangpy as spy

from camera import Camera, DEFAULT_POSITION, DEFAULT_TARGET
from renderer import Lighting, PathTracer, create_device
from sharc import SharcSettings
from dsharc import DSharcSettings
from scene import Scene, demo_scene, load_scene


class App:
    def __init__(self, options):
        self.options = options
        self.device = create_device(options.backend, options.debug)
        data = load_scene(options.scene) if options.scene else demo_scene()
        data.positions *= options.scene_scale
        self.scene = Scene(self.device, data)
        if options.camera_position:
            camera = Camera(options.camera_position, options.camera_target, options.fov)
        elif options.scene:
            low, high = data.bounds
            center = (low + high) * .5
            radius = max(float(np.linalg.norm(high - low)), .01)
            camera = Camera(center + radius * np.array([.65, .4, .85]), center, options.fov)
        else:
            camera = Camera(np.array(DEFAULT_POSITION) * options.scene_scale,
                np.array(DEFAULT_TARGET) * options.scene_scale, options.fov)
        lighting = Lighting(sun_azimuth=options.sun_azimuth, sun_elevation=options.sun_elevation,
            sun_intensity=options.sun_intensity, sun_radius=options.sun_radius,
            sky_intensity=options.sky_intensity)
        self.renderer = PathTracer(self.device, self.scene, camera, lighting,
            options.width, options.height, options.max_bounces, options.spp, options.seed,
            mode=options.renderer, sharc_settings=SharcSettings(options.sharc_capacity,
                options.sharc_downscale, options.sharc_grid_scale, options.sharc_history,
                options.sharc_radiance_scale), dsharc_settings=DSharcSettings(
                    capacity=options.dsharc_capacity, cell_size=options.dsharc_cell_size,
                    history=options.dsharc_history))
        self.renderer.exposure = options.exposure
        self.keys = set()
        self.dragging = False
        self.last_mouse = None
        self.capture_requested = False
        low, high = data.bounds
        self.move_speed = max(float(np.linalg.norm(high - low)) * .10, .01)
        print(f"Scene: {len(data.triangles):,} triangles | {options.backend} | "
              f"{options.width} x {options.height}", flush=True)

    def run(self):
        try:
            if self.options.headless:
                self.run_headless()
            else:
                self.run_interactive()
        finally:
            self.device.wait()

    def run_headless(self):
        start = time.perf_counter()
        for frame in range(self.options.frames):
            self.renderer.frame()
            # Bound command submission backlog in long batch renders.
            if (frame + 1) % 32 == 0:
                self.device.wait()
            if (frame + 1) % max(self.options.frames // 8, 1) == 0:
                print(f"Render {frame + 1}/{self.options.frames} | {self.renderer.sample_count} spp", flush=True)
        self.renderer.save(self.options.output, self.options.linear_output)
        image = self.renderer.linear_image()
        if not np.isfinite(image).all() or np.any(image < 0):
            raise RuntimeError("Nonfinite or negative linear radiance")
        print(f"Saved {self.options.output.resolve()} | {self.renderer.sample_count} spp | "
              f"{time.perf_counter() - start:.2f}s | mean linear RGB {image.mean():.5f}", flush=True)
        if self.renderer.mode != "reference":
            print(f"{self.renderer.mode}: {self.renderer.cache_statistics()}", flush=True)

    def _slider(self, parent, name, attribute, minimum, maximum, target):
        return spy.ui.SliderFloat(parent, name, min=minimum, max=maximum,
            value=getattr(target, attribute), callback=lambda value: setattr(target, attribute, value))

    def setup_ui(self):
        self.ui = spy.ui.Context(self.device)
        panel = spy.ui.Window(self.ui.screen, "Path tracer", spy.float2(12, 12), spy.float2(330, 445))
        spy.ui.Text(panel, "Slang / hardware ray queries / Lambert")
        self.stats = spy.ui.Text(panel, "Starting...")
        for mode in ("reference", "sharc", "dsharc"):
            spy.ui.Button(panel, mode.upper(),
                callback=lambda mode=mode: setattr(self.renderer, "mode", mode))
        self._slider(panel, "Sun azimuth", "sun_azimuth", -180, 180, self.renderer.lighting)
        self._slider(panel, "Sun elevation", "sun_elevation", -10, 90, self.renderer.lighting)
        self._slider(panel, "Sun intensity", "sun_intensity", 0, 12, self.renderer.lighting)
        self._slider(panel, "Sun radius (deg)", "sun_radius", .05, 10, self.renderer.lighting)
        self._slider(panel, "Sky intensity", "sky_intensity", 0, 5, self.renderer.lighting)
        self._slider(panel, "Exposure (EV)", "exposure", -5, 5, self.renderer)
        spy.ui.SliderInt(panel, "Bounces", min=1, max=32, value=self.renderer.max_bounces,
            callback=lambda value: setattr(self.renderer, "max_bounces", value))
        spy.ui.Button(panel, "Reset image + cache", callback=self.renderer.reset)
        spy.ui.Button(panel, "Reset image (keep cache)", callback=self.renderer.reset_image)
        spy.ui.Button(panel, "Save PNG", callback=lambda: setattr(self, "capture_requested", True))
        spy.ui.Text(panel, "RMB: look | WASD: move | Q/E: down/up")
        spy.ui.Text(panel, "Shift: faster | R: reset | F2: save | Esc: quit")

    def keyboard_event(self, event):
        consumed = self.ui.handle_keyboard_event(event)
        # Always release keys, including when a UI widget has acquired focus.
        if event.is_key_release():
            self.keys.discard(event.key)
        if event.is_key_press() and not consumed:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.r:
                self.renderer.reset()
            elif event.key == spy.KeyCode.f2:
                self.capture_requested = True
            self.keys.add(event.key)

    def mouse_event(self, event):
        consumed = self.ui.handle_mouse_event(event)
        if event.is_button_up() and event.button == spy.MouseButton.right:
            self.dragging = False
        if event.is_button_down() and event.button == spy.MouseButton.right and not consumed:
            self.dragging = True
        position = (event.pos.x, event.pos.y)
        if event.is_move() and self.dragging and self.last_mouse is not None:
            self.renderer.camera.rotate(position[0] - self.last_mouse[0], position[1] - self.last_mouse[1])
        self.last_mouse = position

    def resize(self, width, height):
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height,
                format=self.surface_format, vsync=self.options.vsync)
            self.renderer.resize(width, height)
        else:
            self.surface.unconfigure()

    def run_interactive(self):
        mode = spy.WindowMode.minimized if self.options.window_frames else spy.WindowMode.normal
        self.window = spy.Window(width=self.options.width, height=self.options.height,
            title="DSHARC sample | Slang path tracer", resizable=True, mode=mode)
        self.surface = self.device.create_surface(self.window)
        self.surface_format = next((format for format in (spy.Format.rgba8_unorm, spy.Format.bgra8_unorm)
            if format in self.surface.info.formats), None)
        if self.surface_format is None:
            raise RuntimeError("This sample requires an 8-bit UNORM presentation surface")
        self.surface.configure(width=self.options.width, height=self.options.height,
            format=self.surface_format, vsync=self.options.vsync)
        self.setup_ui()
        self.window.on_keyboard_event = self.keyboard_event
        self.window.on_mouse_event = self.mouse_event
        self.window.on_resize = self.resize
        previous = time.perf_counter()
        frames = 0
        while not self.window.should_close():
            now = time.perf_counter()
            dt, previous = min(now - previous, .1), now
            self.window.process_events()
            k = spy.KeyCode
            self.renderer.camera.move(int(k.d in self.keys) - int(k.a in self.keys),
                int(k.e in self.keys) - int(k.q in self.keys),
                int(k.w in self.keys) - int(k.s in self.keys),
                dt * self.move_speed * (4 if k.left_shift in self.keys else 1))
            if not self.surface.config:
                time.sleep(.02)
                continue
            texture = self.surface.acquire_next_image()
            if texture is None:
                time.sleep(.01)
                continue
            self.renderer.resize(texture.width, texture.height)
            encoder = self.device.create_command_encoder()
            self.renderer.render(encoder)
            # The target is UNORM, not sRGB: display already contains encoded RGB.
            encoder.blit(texture, self.renderer.display)
            self.stats.text = f"{self.renderer.mode} | {self.renderer.sample_count} spp | {dt * 1000:.1f} ms"
            self.ui.begin_frame(self.window.width, self.window.height)
            self.ui.end_frame(texture, encoder)
            self.device.submit_command_buffer(encoder.finish())
            del texture
            self.surface.present()
            if self.capture_requested:
                self.renderer.save(self.options.output, self.options.linear_output)
                print(f"Saved {self.options.output.resolve()}", flush=True)
                self.capture_requested = False
            frames += 1
            if self.options.window_frames and frames >= self.options.window_frames:
                self.window.close()
        self.device.wait()
        self.surface.unconfigure()

