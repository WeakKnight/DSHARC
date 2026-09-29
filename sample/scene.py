"""Small static triangle scene and SlangPy BLAS/TLAS construction.

The built-in scene needs no assets. Imported meshes use diffuse material factors
or vertex/face colors; textures, alpha, transmission and animation are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import numpy as np
import slangpy as spy
import trimesh


@dataclass
class SceneData:
    positions: np.ndarray
    normals: np.ndarray
    triangles: np.ndarray
    albedos: np.ndarray
    emissions: np.ndarray

    def __post_init__(self):
        for name, columns, dtype in (
            ("positions", 3, np.float32), ("normals", 3, np.float32),
            ("triangles", 3, np.uint32), ("albedos", 4, np.float32),
            ("emissions", 4, np.float32),
        ):
            value = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            if value.ndim != 2 or value.shape[1] != columns or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} array")
            setattr(self, name, value)
        if len(self.triangles) == 0 or len(self.positions) == 0:
            raise ValueError("Scene must contain at least one triangle")
        if self.triangles.max() >= len(self.positions) or len(self.normals) != len(self.positions):
            raise ValueError("Scene vertex/index layout is inconsistent")
        if len(self.albedos) != len(self.triangles) or len(self.emissions) != len(self.triangles):
            raise ValueError("Expected one albedo/emission per triangle")
        self.albedos[:, :3] = np.clip(self.albedos[:, :3], 0, 1)
        self.emissions[:, :3] = np.maximum(self.emissions[:, :3], 0)

    @property
    def bounds(self):
        return self.positions.min(axis=0), self.positions.max(axis=0)


class SceneBuilder:
    def __init__(self):
        self.positions, self.normals, self.triangles = [], [], []
        self.albedos, self.emissions = [], []
        self.vertex_count = 0

    def add(self, mesh: trimesh.Trimesh, color, emission=(0, 0, 0), smooth=False):
        mesh = mesh.copy()
        if not smooth:
            mesh.unmerge_vertices()
        positions = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        if not len(faces):
            return
        edge0 = positions[faces[:, 1]] - positions[faces[:, 0]]
        edge1 = positions[faces[:, 2]] - positions[faces[:, 0]]
        keep = np.linalg.norm(np.cross(edge0, edge1), axis=1) > 1e-12
        face_colors = np.broadcast_to(np.asarray(color, dtype=np.float32), (len(faces), 3))[keep]
        face_emission = np.broadcast_to(np.asarray(emission, dtype=np.float32), (len(faces), 3))[keep]
        faces = faces[keep]
        if not len(faces):
            return
        # Area-weighted smooth normals, or exact face normals after unmerging.
        # NumPy accumulation keeps SciPy an optional dependency of trimesh.
        normals = np.zeros_like(positions)
        face_normals = np.cross(edge0, edge1)[keep]
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
        self.positions.append(positions)
        self.normals.append(normals)
        self.triangles.append(faces + self.vertex_count)
        self.albedos.append(np.column_stack((face_colors, np.ones(len(faces)))))
        self.emissions.append(np.column_stack((face_emission, np.zeros(len(faces)))))
        self.vertex_count += len(positions)

    def box(self, size, center, color):
        mesh = trimesh.creation.box(extents=size)
        mesh.apply_translation(center)
        self.add(mesh, color)

    def finish(self):
        if not self.triangles:
            raise ValueError("Scene has no nondegenerate triangle meshes")
        return SceneData(*(np.concatenate(value) for value in (
            self.positions, self.normals, self.triangles, self.albedos, self.emissions)))


def demo_scene() -> SceneData:
    """A roofed room lit through a side window and a small rear-facing doorway."""
    builder = SceneBuilder()
    plaster = (.72, .69, .62)
    wood = (.43, .27, .13)
    # Interior: 8 x 8 meters, 3.4 m high. Thick overlapping shell pieces
    # avoid cracks; the window and doorway are actual openings, without glass.
    builder.box((16, .2, 16), (0, -.2, 0), (.34, .32, .26))
    builder.box((8.4, .18, 8.4), (0, -.10, 0), wood)
    for i in range(16):
        tint = 1.0 + .035 * ((i * 7) % 5 - 2)
        builder.box((.494, .04, 8), (-3.75 + i * .5, -.02, 0),
            np.array([.48, .35, .21]) * tint)
    builder.box((8.4, .22, 8.4), (0, 3.51, 0), (.76, .74, .69))
    builder.box((8.4, 3.4, .22), (0, 1.7, -4.11), plaster)
    builder.box((.22, 3.4, 8.4), (4.11, 1.7, 0), (.29, .47, .38))

    # Left-wall window: z=[-2.2, .8], y=[1.0, 2.7].
    builder.box((.22, 3.4, 1.8), (-4.11, 1.7, -3.1), plaster)
    builder.box((.22, 3.4, 3.2), (-4.11, 1.7, 2.4), plaster)
    builder.box((.22, 1.0, 3.0), (-4.11, .5, -.7), plaster)
    builder.box((.22, .7, 3.0), (-4.11, 3.05, -.7), plaster)
    builder.box((.42, .10, 3.25), (-4.02, .98, -.7), wood)
    builder.box((.26, 1.7, .075), (-4.11, 1.85, -.7), wood)
    builder.box((.26, .075, 3.0), (-4.11, 1.85, -.7), wood)

    # Front wall behind the camera: only a 1.6 x 2.35 m doorway admits light.
    builder.box((3.2, 3.4, .22), (-2.4, 1.7, 4.11), plaster)
    builder.box((3.2, 3.4, .22), (2.4, 1.7, 4.11), plaster)
    builder.box((1.6, 1.05, .22), (0, 2.875, 4.11), plaster)

    # Furniture creates shadowed receivers and warm reflected light.
    builder.box((1.25, .12, 2.5), (-3.0, .88, -.6), wood)
    for x in (-3.48, -2.52):
        for z in (-1.6, .4):
            builder.box((.12, .82, .12), (x, .41, z), wood)
    builder.box((1.4, 1.25, 1.35), (-.85, .625, -1.55), (.64, .20, .075))
    builder.box((2.2, .65, .85), (2.3, .325, -3.35), wood)
    for y in (1.45, 2.25):
        builder.box((2.4, .10, .55), (2.3, y, -3.68), wood)
    for x, height, color in ((1.5, .40, (.38, .12, .08)),
                              (1.8, .50, (.15, .27, .35)),
                              (2.1, .34, (.55, .42, .19))):
        builder.box((.18, height, .30), (x, 1.50 + height * .5, -3.64), color)
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=.82)
    sphere.apply_translation((1.1, .82, .1))
    builder.add(sphere, (.09, .28, .56), smooth=True)
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=.30)
    sphere.apply_translation((-2.9, 1.24, -.65))
    builder.add(sphere, (.73, .69, .57), smooth=True)
    return builder.finish()


def load_scene(path: Path) -> SceneData:
    if not path.is_file():
        raise FileNotFoundError(path)
    source = trimesh.load_scene(str(path), process=False)
    builder = SceneBuilder()
    ignored_textures = False
    for node in source.graph.nodes_geometry:
        transform, geometry_name = source.graph[node]
        geometry = source.geometry[geometry_name]
        if not isinstance(geometry, trimesh.Trimesh):
            continue
        mesh = geometry.copy()
        mesh.apply_transform(transform)
        color, emission = np.array([.65, .65, .65]), np.zeros(3)
        material = getattr(mesh.visual, "material", None)
        if material is not None:
            factor = getattr(material, "baseColorFactor", None)
            if factor is not None:
                # trimesh stores the glTF linear factor as an 8-bit array.
                color = np.asarray(factor, dtype=np.float32)[:3] / 255.0
            else:
                factor = getattr(material, "diffuse", None)
                if factor is not None:
                    color = np.asarray(factor, dtype=np.float32)[:3] / 255.0
            factor = getattr(material, "emissiveFactor", None)
            if factor is not None:
                emission = np.asarray(factor, dtype=np.float32)[:3]
            ignored_textures |= (getattr(material, "baseColorTexture", None) is not None or
                                 getattr(material, "image", None) is not None)
        elif mesh.visual.kind in ("vertex", "face"):
            srgb = np.asarray(mesh.visual.face_colors, dtype=np.float32)[:, :3] / 255.0
            color = np.where(srgb <= .04045, srgb / 12.92, ((srgb + .055) / 1.055) ** 2.4)
        builder.add(mesh, color, emission, smooth=True)
    if ignored_textures:
        warnings.warn("Sample imports material factors only; texture maps are not sampled.", stacklevel=2)
    return builder.finish()


class Scene:
    def __init__(self, device: spy.Device, data: SceneData):
        self.device, self.data = device, data
        self.buffers = {
            name: device.create_buffer(usage=spy.BufferUsage.shader_resource,
                data=getattr(data, name), label=f"sample.{name}")
            for name in ("positions", "normals", "triangles", "albedos", "emissions")
        }
        triangles = spy.AccelerationStructureBuildInputTriangles({
            "vertex_buffers": [{"buffer": self.buffers["positions"]}],
            "vertex_format": spy.Format.rgb32_float,
            "vertex_count": len(data.positions), "vertex_stride": 12,
            "index_buffer": {"buffer": self.buffers["triangles"]},
            "index_format": spy.IndexFormat.uint32, "index_count": data.triangles.size,
            "flags": spy.AccelerationStructureGeometryFlags.opaque,
        })
        self.blas = self._build(spy.AccelerationStructureBuildDesc({"inputs": [triangles]}), "blas")
        instances = device.create_acceleration_structure_instance_list(size=1)
        instances.write(0, {
            "transform": spy.float3x4(np.eye(4, dtype=np.float32)[:3]),
            "instance_id": 0, "instance_mask": 0xff,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": self.blas.handle,
        })
        self.tlas = self._build(spy.AccelerationStructureBuildDesc({
            "inputs": [instances.build_input_instances()]}), "tlas")
        low, high = data.bounds
        self.epsilon = max(float(np.linalg.norm(high - low)) * 1e-6, 1e-6)

    def _build(self, description, label):
        sizes = self.device.get_acceleration_structure_sizes(description)
        scratch = self.device.create_buffer(size=sizes.scratch_size,
            usage=spy.BufferUsage.unordered_access, label=f"sample.{label}.scratch")
        acceleration = self.device.create_acceleration_structure(
            size=sizes.acceleration_structure_size, label=f"sample.{label}")
        encoder = self.device.create_command_encoder()
        encoder.build_acceleration_structure(desc=description, dst=acceleration, src=None, scratch_buffer=scratch)
        self.device.submit_command_buffer(encoder.finish())
        # Explicitly retain scratch/input lifetimes until this startup build completes.
        self.device.wait()
        return acceleration

    def bind(self, cursor: spy.ShaderCursor):
        cursor.tlas = self.tlas
        for name, buffer in self.buffers.items():
            cursor[name] = buffer
        cursor.ray_epsilon = self.epsilon
