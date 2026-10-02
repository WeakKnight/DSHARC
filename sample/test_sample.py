"""CPU validation and actual GPU transport/accumulation tests (no image mocks)."""
from __future__ import annotations

import argparse
import contextlib
import io
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import trimesh

from camera import Camera
from entry_point import parse_args
from renderer import Lighting, PathTracer, create_device
from scene import Scene, SceneBuilder, demo_scene, load_scene

BACKEND = "d3d12"


def plane_scene(blocker=False):
    builder = SceneBuilder()
    plane = trimesh.Trimesh(vertices=[[-20, 0, -20], [-20, 0, 20], [20, 0, 20], [20, 0, -20]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False)
    builder.add(plane, (.5, .5, .5))
    if blocker:
        # Outside the camera frustum, but between the receiver and the sun.
        builder.box((1.6, 1.6, 1.6), (2, 2, 0), (.5, .5, .5))
    return builder.finish()


class CPUChecks(unittest.TestCase):
    def test_geometry_and_camera(self):
        data = demo_scene()
        self.assertGreater(len(data.triangles), 1000)
        np.testing.assert_allclose(np.linalg.norm(data.normals, axis=1), 1, atol=1e-5)
        camera = Camera()
        basis = np.stack(camera.basis())
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-10)

    def test_cli_rejects_invalid_settings(self):
        for arguments in (["--width", "0"], ["--sun-radius", "0"], ["--sky-intensity", "nan"],
                          ["--sun-intensity", "-1"], ["--camera-position", "0", "1", "2"],
                          ["--sharc-grid-scale", "nan"], ["--sharc-capacity", "1234"],
                          ["--sharc-downscale", "0"], ["--sharc-radiance-scale", "0"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(arguments)

    def test_gltf_scene_graph_import(self):
        mesh = trimesh.creation.box(extents=(1, 1, 1))
        scene = trimesh.Scene()
        scene.add_geometry(mesh, node_name="left")
        matrix = np.eye(4)
        matrix[:3, 3] = [3, 0, 0]
        scene.add_geometry(mesh, node_name="right", transform=matrix)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scene.glb"
            path.write_bytes(scene.export(file_type="glb"))
            loaded = load_scene(path)
            self.assertEqual(len(loaded.triangles), 24)
            np.testing.assert_allclose(loaded.bounds[0], [-.5, -.5, -.5])
            np.testing.assert_allclose(loaded.bounds[1], [3.5, .5, .5])


class GPUChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device(BACKEND)
        cls.plane = Scene(cls.device, plane_scene())
        cls.blocked = Scene(cls.device, plane_scene(blocker=True))
        cls.courtyard = Scene(cls.device, demo_scene())

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def renderer(self, lighting=None, scene=None, width=17, height=13, bounces=1, spp=8):
        camera = Camera((0, 4, 0), (0, 0, 0), 12)
        return PathTracer(self.device, scene or self.plane, camera, lighting or Lighting(),
            width, height, bounces, spp, seed=19)

    def render(self, renderer, frames=1):
        for _ in range(frames):
            renderer.frame()
        value = renderer.linear_image()
        self.assertTrue(np.isfinite(value).all())
        self.assertGreaterEqual(float(value.min()), 0)
        return value

    def test_all_lights_off_is_black(self):
        renderer = self.renderer(Lighting(sun_intensity=0, sky_intensity=0), self.courtyard, bounces=6)
        np.testing.assert_array_equal(self.render(renderer), 0)

    def test_constant_sky_lambert_energy_and_last_segment(self):
        light = Lighting(sun_intensity=0, sky_intensity=1,
            sky_zenith=(1, 1, 1), sky_horizon=(1, 1, 1), sky_ground=(1, 1, 1))
        renderer = self.renderer(light)
        # A .5 Lambert reflector under uniform unit incident radiance returns .5.
        np.testing.assert_allclose(self.render(renderer), .5, atol=2e-6)
        self.assertEqual(renderer.sample_count, 8)

    def test_solar_disk_mis_preserves_energy(self):
        light = Lighting(sun_elevation=90, sun_color=(1, 1, 1), sun_intensity=math.pi,
            sun_radius=20, sky_intensity=0)
        renderer = self.renderer(light, spp=32)
        value = self.render(renderer, 16)
        # Large disk makes accidental NEE/BSDF double counting easy to detect.
        self.assertAlmostEqual(float(value.mean()), .5, delta=.008)

    def test_primary_ray_sees_solar_disk(self):
        light = Lighting(sun_elevation=90, sun_color=(1, 1, 1), sun_intensity=math.pi,
            sun_radius=20, sky_intensity=0)
        renderer = self.renderer(light)
        renderer.camera = Camera((0, 4, 0), (0, 5, 0), 5)
        expected = 1.0 / math.sin(math.radians(20)) ** 2
        np.testing.assert_allclose(self.render(renderer), expected, rtol=1e-5)

    def test_ray_traced_sun_shadow(self):
        light = Lighting(sun_azimuth=90, sun_elevation=45, sun_color=(1, 1, 1),
            sun_intensity=math.pi, sun_radius=.26785, sky_intensity=0)
        clear = self.render(self.renderer(light), 4)
        blocked = self.render(self.renderer(light, self.blocked), 4)
        self.assertGreater(float(clear.mean()), .3)
        self.assertLess(float(blocked[4:9, 6:11].mean()), .0001)

    def test_accumulation_reset_batching_and_export(self):
        renderer = self.renderer(width=31, height=23, spp=1)
        first = self.render(renderer)
        renderer.samples_per_frame = 3
        batched = self.render(renderer)
        self.assertEqual(renderer.sample_count, 4)
        renderer.reset()
        renderer.samples_per_frame = 4
        np.testing.assert_allclose(self.render(renderer), batched, rtol=2e-6, atol=2e-6)
        renderer.samples_per_frame = 1
        renderer.reset()
        np.testing.assert_array_equal(self.render(renderer), first)
        renderer.exposure += 1
        self.render(renderer)
        self.assertEqual(renderer.sample_count, 2)  # display changes keep linear history
        renderer.lighting.sky_intensity += .2
        self.render(renderer)
        self.assertEqual(renderer.sample_count, 1)
        renderer.camera.position[0] += .1
        self.render(renderer)
        self.assertEqual(renderer.sample_count, 1)
        renderer.resize(19, 11)
        resized = self.render(renderer)
        self.assertEqual(resized.shape, (11, 19, 3))
        with tempfile.TemporaryDirectory() as directory:
            png, linear = Path(directory) / "display.png", Path(directory) / "linear.npy"
            renderer.save(png, linear)
            with Image.open(png) as image:
                self.assertEqual(image.size, (19, 11))
            np.testing.assert_array_equal(np.load(linear), resized)

    def test_additional_bounces_add_indirect_illumination(self):
        renderer = PathTracer(self.device, self.courtyard, Camera(), Lighting(),
            64, 48, max_bounces=1, samples_per_frame=16)
        direct = self.render(renderer, 4)
        renderer.max_bounces = 6
        indirect = self.render(renderer, 4)
        self.assertGreater(float(indirect.mean()), float(direct.mean()) + .005)
        self.assertGreaterEqual(float((indirect - direct).min()), -1e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("d3d12", "vulkan", "metal"), default="d3d12")
    options = parser.parse_args()
    BACKEND = options.backend
    unittest.main(argv=[__file__], verbosity=2)
