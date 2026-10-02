"""Real GPU tests for demand requests, frozen feedback, transport and fallbacks."""
import argparse
import unittest
import numpy as np
import trimesh
from camera import Camera
from renderer import Lighting, PathTracer, create_device
from scene import Scene, SceneBuilder, demo_scene
from dsharc import DSharcSettings
from compare_sharc import metrics

BACKEND = "d3d12"


class DSharcChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device(BACKEND)
        cls.room = Scene(cls.device, demo_scene())

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def tracer(self, scene=None, camera=None, lighting=None, **kwargs):
        return PathTracer(self.device, scene or self.room, camera or Camera(), lighting or Lighting(),
            width=64, height=48, max_bounces=32, samples_per_frame=8, mode="dsharc", **kwargs)

    def render(self, renderer, frames=1):
        for i in range(frames):
            renderer.frame()
            if i % 32 == 31:
                self.device.wait()
        value = renderer.linear_image()
        self.assertTrue(np.isfinite(value).all())
        self.assertGreaterEqual(float(value.min()), 0)
        return value

    def test_furnace_energy_and_feedback(self):
        builder = SceneBuilder()
        builder.add(trimesh.creation.box(extents=(4,4,4)), (.2,.5,.8), (.8,.5,.2))
        r = self.tracer(Scene(self.device, builder.finish()), Camera((0,0,0),(0,0,-1)),
            Lighting(sun_intensity=0, sky_intensity=0),
            dsharc_settings=DSharcSettings(cell_size=.25))
        value = self.render(r, 128)
        np.testing.assert_allclose(value.mean((0,1)), 1, atol=.015)
        self.assertGreater(r.cache_statistics()["requested_path_fraction"], .8)
        self.assertGreater(r.cache_statistics()["feedback_hits"], 100)
        self.assertEqual(r.cache_statistics()["gather_fallbacks"], 0)

    def test_room_bias_and_full_hash_fallback(self):
        for capacity in (1 << 20, 1024):
            r = self.tracer(dsharc_settings=DSharcSettings(capacity=capacity))
            r.mode = "reference"
            reference = self.render(r, 256)
            r.mode = "dsharc"
            value = self.render(r, 256)
            diff = metrics(value, reference)
            self.assertLess(abs(diff["mean_bias_percent"]), 1.5)
            self.assertLess(diff["block8_relative_l1_percent"], 3)
            stats = r.cache_statistics()
            if capacity == 1024:
                self.assertGreater(stats["failed_requests"], 0)
            else:
                self.assertGreater(stats["requested_path_fraction"], .85)

    def test_empty_active_list(self):
        builder = SceneBuilder()
        builder.box((100, .1, 100), (0,-2,0), (.5,.5,.5))
        r = self.tracer(Scene(self.device, builder.finish()), Camera((0,0,0),(0,1,0),30),
            Lighting(sun_intensity=0, sky_intensity=1,
                sky_zenith=(1,1,1), sky_horizon=(1,1,1), sky_ground=(1,1,1)))
        np.testing.assert_allclose(self.render(r, 4), 1, atol=1e-6)
        self.assertEqual(r.cache_statistics()["active_entries"], 0)

    def test_resets_batch_resize_and_mode_switch(self):
        r = self.tracer()
        self.render(r, 16)
        self.assertGreater(r.cache_statistics()["active_entries"], 0)
        r.reset_image()
        self.render(r)
        self.assertEqual(r.demand_cache.frame_index, 17)
        self.assertEqual(r.sample_count, 8)
        r.lighting.sun_intensity = r.lighting.sky_intensity = 0
        np.testing.assert_array_equal(self.render(r, 4), 0)
        self.assertEqual(r.demand_cache.frame_index, 4)
        r.camera.position[0] += .1
        self.render(r)
        self.assertEqual(r.demand_cache.frame_index, 1)
        r.dsharc_settings.cell_size *= 2
        self.render(r)
        self.assertEqual(r.demand_cache.frame_index, 1)
        r.samples_per_frame = 3
        self.render(r)
        self.assertEqual(r.demand_cache.path_count, 64*48*3)
        self.assertEqual(r.demand_cache.frame_index, 2)
        r.resize(65,49)
        self.assertEqual(self.render(r).shape, (49,65,3))
        self.assertEqual(r.demand_cache.frame_index, 1)
        r.mode = "reference"
        np.testing.assert_array_equal(self.render(r), 0)
        r.mode = "dsharc"
        np.testing.assert_array_equal(self.render(r), 0)
        self.assertEqual(r.demand_cache.frame_index, 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=("d3d12", "vulkan", "metal"), default="d3d12")
    BACKEND = p.parse_args().backend
    unittest.main(argv=[__file__], verbosity=2)
