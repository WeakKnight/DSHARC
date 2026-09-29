"""GPU correctness checks for the real SDK, including paths that actually query it."""
import argparse
import unittest
import numpy as np
import trimesh
from camera import Camera
from renderer import Lighting, PathTracer, create_device
from scene import Scene, SceneBuilder, demo_scene
from sharc import SharcSettings
from compare_sharc import metrics

BACKEND = "d3d12"


class SharcChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device(BACKEND)
        cls.room = Scene(cls.device, demo_scene())

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def render(self, renderer, frames):
        for i in range(frames):
            renderer.frame()
            if i % 32 == 31:
                self.device.wait()
        result = renderer.linear_image()
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(float(result.min()), 0)
        return result

    def tracer(self, scene=None, camera=None, lighting=None, **kwargs):
        return PathTracer(self.device, scene or self.room, camera or Camera(), lighting or Lighting(),
            width=96, height=64, max_bounces=32, samples_per_frame=16, mode="sharc",
            sharc_settings=SharcSettings(capacity=1 << 20), **kwargs)

    def test_emissive_closed_furnace_rgb_energy(self):
        # L = E + rho*L = 1 for each channel despite very different albedos.
        # This catches double albedo, missing tails, and double/missing emission.
        builder = SceneBuilder()
        builder.add(trimesh.creation.box(extents=(4, 4, 4)), (.2, .5, .8), (.8, .5, .2))
        renderer = self.tracer(Scene(self.device, builder.finish()), Camera((0, 0, 0), (0, 0, -1)),
            Lighting(sky_intensity=0, sun_intensity=0))
        renderer.sharc_settings.scene_scale = 8
        result = self.render(renderer, 128)
        np.testing.assert_allclose(result.mean((0, 1)), [1, 1, 1], atol=.015)
        self.assertLess(float(np.abs(result - 1).mean()), .02)
        self.assertGreater(renderer.cache_statistics()["terminated_path_fraction"], .8)
        self.assertGreater(renderer.cache_statistics()["resampled_update_fraction"], .5)

    def test_room_bias_and_cache_coverage(self):
        renderer = self.tracer(seed=51)
        renderer.mode = "reference"
        reference = self.render(renderer, 256)
        renderer.mode = "sharc"
        renderer.seed = 131
        cached = self.render(renderer, 256)
        difference = metrics(cached, reference)
        self.assertLess(abs(difference["mean_bias_percent"]), 1.0)
        self.assertLess(difference["block8_relative_l1_percent"], 2.0)
        self.assertGreater(renderer.cache_statistics()["terminated_path_fraction"], .70)
        self.assertGreater(renderer.cache_statistics()["resampled_update_fraction"], .5)

    def test_invalidation_fallback_resize_and_odd_tiles(self):
        renderer = self.tracer()
        self.render(renderer, 48)
        self.assertGreater(renderer.cache.occupancy(), 0)
        renderer.exposure += 1
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 49)
        sequence_end = renderer.sample_count
        renderer.reset_image()
        self.assertEqual(renderer.cache.frame_index, 49)
        self.assertEqual(renderer.sample_count, 0)
        self.assertEqual(renderer._sample_offset, sequence_end)
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 50)
        self.assertEqual(renderer.sample_count, renderer.samples_per_frame)
        # A populated, illuminated cache must not leak stale energy when dark.
        renderer.lighting.sun_intensity = renderer.lighting.sky_intensity = 0
        np.testing.assert_array_equal(self.render(renderer, 8), 0)
        self.assertEqual(renderer.cache.frame_index, 8)
        renderer.camera.position[0] += .1
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 1)
        renderer.sharc_settings.scene_scale = 40
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 1)
        renderer.resize(97, 63)
        self.assertEqual(self.render(renderer, 1).shape, (63, 97, 3))
        self.assertEqual(renderer.cache.frame_index, 1)
        renderer.mode = "reference"
        np.testing.assert_array_equal(self.render(renderer, 1), 0)
        renderer.mode = "sharc"
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 1)
        renderer.reset()
        self.render(renderer, 1)
        self.assertEqual(renderer.cache.frame_index, 1)

    def test_black_albedo_and_zero_lighting(self):
        builder = SceneBuilder()
        builder.add(trimesh.creation.box(extents=(4, 4, 4)), (0, 0, 0))
        renderer = self.tracer(Scene(self.device, builder.finish()), Camera((0, 0, 0), (0, 0, -1)))
        np.testing.assert_array_equal(self.render(renderer, 16), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("d3d12", "vulkan"), default="d3d12")
    BACKEND = parser.parse_args().backend
    unittest.main(argv=[__file__], verbosity=2)
