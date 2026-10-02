"""GPU checks for exact keys, concurrent insertion, holes and busy-slot fallback."""
import argparse
from pathlib import Path
import unittest

import numpy as np
import slangpy as spy

from renderer import create_device

BACKEND = "metal"
INVALID = 0xffffffff
CAPACITY = 64

SOURCE = r'''
#include "DSharcHashGrid.h"
uniform DSharcParameters g_cache;
StructuredBuffer<uint2> g_input;
RWStructuredBuffer<uint> g_output;
uniform uint g_count;

[shader("compute")]
[numthreads(64, 1, 1)]
void insert_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_count) return;
    uint2 words = g_input[tid.x];
    uint64_t key = uint64_t(words.x) | (uint64_t(words.y) << 32);
    g_output[tid.x] = DSharcFindOrInsertKey(g_cache, key);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void find_main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= g_count) return;
    uint2 words = g_input[tid.x];
    g_output[tid.x] = DSharcFindKey(g_cache,
        uint64_t(words.x) | (uint64_t(words.y) << 32));
}
'''


def key_hash(key):
    h = ((key & INVALID) ^ ((key >> 32) * 0x9e3779b9)) & INVALID
    h = ((h ^ (h >> 16)) * 0x7feb352d) & INVALID
    h = ((h ^ (h >> 15)) * 0x846ca68b) & INVALID
    return h ^ (h >> 16)


class HashChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = create_device(BACKEND)
        cls.session = cls.device.create_slang_session(compiler_options={
            "include_paths": [Path(__file__).resolve().parents[1] / "include"],
            "defines": {"DSHARC_SPLIT_KEY_ATOMICS": "1" if BACKEND == "metal" else "0"},
            **({"capabilities": ["metallib_4_0"]} if BACKEND == "metal" else {}),
        })
        module = cls.session.load_module_from_source("dsharc_hash_checks", SOURCE)
        cls.kernels = {name: cls.device.create_compute_kernel(
            cls.session.link_program([module], [module.entry_point(name + "_main")]))
            for name in ("insert", "find")}

    @classmethod
    def tearDownClass(cls):
        cls.device.wait()

    def buffer(self, values, stride):
        return self.device.create_buffer(data=values, size=values.nbytes, struct_size=stride,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)

    def setUp(self):
        self.keys = self.buffer(np.zeros((CAPACITY, 2), np.uint32), 8)
        self.states = self.buffer(np.zeros(CAPACITY, np.uint32), 4)

    def dispatch(self, values, name="insert"):
        words = np.array([(v & INVALID, v >> 32) for v in values], np.uint32)
        inputs = self.buffer(words, 8)
        outputs = self.buffer(np.full(len(values), INVALID, np.uint32), 4)
        cache = {"capacity": CAPACITY, "keys": self.keys}
        if BACKEND == "metal":
            cache["keyStates"] = self.states
        self.kernels[name].dispatch(thread_count=[len(values), 1, 1],
            vars={"g_cache": cache, "g_input": inputs, "g_output": outputs, "g_count": len(values)})
        self.device.wait()
        return outputs.to_numpy().view(np.uint32).ravel()

    def stored(self):
        return self.keys.to_numpy().view(np.uint64).ravel()

    def test_concurrent_duplicate_and_exact_high_word(self):
        a, b = (1 << 62) | 7, (1 << 62) | (1 << 32) | 7
        for _ in range(16):
            self.keys.copy_from_numpy(np.zeros((CAPACITY, 2), np.uint32))
            self.states.copy_from_numpy(np.zeros(CAPACITY, np.uint32))
            values = [a, b] * 2048
            result = self.dispatch(values)
            stored = self.stored()
            if BACKEND == "metal":
                states = self.states.to_numpy().view(np.uint32).ravel()
                self.assertFalse(np.any(states == 1))
            self.assertEqual(int(np.count_nonzero(stored == a)), 1)
            self.assertEqual(int(np.count_nonzero(stored == b)), 1)
            for value, index in zip(values, result):
                if index != INVALID:
                    self.assertEqual(int(stored[index]), value)
            retry = self.dispatch([a, b] * 256)
            self.assertTrue(np.all(retry != INVALID))
            self.assertEqual(len(np.unique(retry)), 2)

    def test_hole_does_not_hide_or_duplicate_key(self):
        a = (1 << 62) | 1
        start = key_hash(a) % CAPACITY
        b = next((1 << 62) | v for v in range(2, 10000)
            if key_hash((1 << 62) | v) % CAPACITY == start)
        first = int(self.dispatch([a])[0])
        second = int(self.dispatch([b])[0])
        self.assertNotEqual(first, second)
        words = self.keys.to_numpy().view(np.uint32).reshape(CAPACITY, 2)
        words[first] = 0
        self.keys.copy_from_numpy(words)
        states = self.states.to_numpy().view(np.uint32).ravel()
        states[first] = 0
        self.states.copy_from_numpy(states)
        np.testing.assert_array_equal(self.dispatch([b], "find"), [second])
        np.testing.assert_array_equal(self.dispatch([b] * 4096), second)
        self.assertEqual(int(np.count_nonzero(self.stored() == b)), 1)

    def test_busy_slot_and_full_window_fall_back(self):
        key = (1 << 62) | 123
        start = key_hash(key) % CAPACITY
        if BACKEND == "metal":
            for offset in (0, 1, 15):
                states = np.zeros(CAPACITY, np.uint32)
                states[(start + offset) % CAPACITY] = 1
                self.states.copy_from_numpy(states)
                self.assertTrue(np.all(self.dispatch([key] * 4096) == INVALID))
                np.testing.assert_array_equal(self.stored(), 0)
        words = np.array([(v + 1, 0x40000000) for v in range(CAPACITY)], np.uint32)
        self.keys.copy_from_numpy(words)
        self.states.copy_from_numpy(np.full(CAPACITY, 3, np.uint32))
        self.assertTrue(np.all(self.dispatch([key] * 4096) == INVALID))
        np.testing.assert_array_equal(self.keys.to_numpy().view(np.uint32).reshape(CAPACITY, 2), words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("d3d12", "vulkan", "metal"), default="metal")
    BACKEND = parser.parse_args().backend
    unittest.main(argv=[__file__], verbosity=2)
