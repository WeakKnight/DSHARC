from __future__ import annotations

import math
import numpy as np
import slangpy as spy


DEFAULT_POSITION = (2.8, 1.8, 3.25)
DEFAULT_TARGET = (-.65, 1.35, -1.25)
DEFAULT_FOV = 65.0


class Camera:
    def __init__(self, position=DEFAULT_POSITION, target=DEFAULT_TARGET, fov=DEFAULT_FOV):
        self.position = np.asarray(position, dtype=np.float64).copy()
        delta = np.asarray(target, dtype=np.float64) - self.position
        if not np.isfinite(delta).all() or np.linalg.norm(delta) < 1e-8:
            raise ValueError("Camera position and target must be finite and distinct")
        delta /= np.linalg.norm(delta)
        self.yaw = math.atan2(delta[0], -delta[2])
        self.pitch = math.asin(float(np.clip(delta[1], -1, 1)))
        self.fov = fov

    def basis(self):
        forward = np.array([math.sin(self.yaw) * math.cos(self.pitch), math.sin(self.pitch),
                            -math.cos(self.yaw) * math.cos(self.pitch)])
        right = np.array([math.cos(self.yaw), 0, math.sin(self.yaw)])
        up = np.cross(right, forward)
        return forward, right, up

    def rotate(self, dx, dy):
        self.yaw += float(dx) * .003
        self.pitch = float(np.clip(self.pitch - float(dy) * .003, -1.55, 1.55))

    def move(self, right_amount, up_amount, forward_amount, distance):
        forward, right, _ = self.basis()
        direction = right * right_amount + forward * forward_amount + np.array([0, up_amount, 0])
        length = np.linalg.norm(direction)
        if length > 0:
            self.position += direction / length * distance

    def signature(self):
        return (*self.position, self.yaw, self.pitch, self.fov)

    def shader_values(self):
        forward, right, up = self.basis()
        return {"position": spy.float3(*self.position), "forward": spy.float3(*forward),
                "right": spy.float3(*right), "up": spy.float3(*up),
                "tan_half_fov": math.tan(math.radians(self.fov) * .5)}
