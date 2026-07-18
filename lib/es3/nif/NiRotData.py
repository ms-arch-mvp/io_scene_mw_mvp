from __future__ import annotations

from enum import IntEnum

from es3.utils.math import np, quaternion_from_euler_angle, quaternion_mul, zeros
from .NiFloatData import KeyType, NiFloatData


class AxisOrder(IntEnum):
    XYZ = 0
    XZY = 1
    YZX = 2
    YXZ = 3
    ZXY = 4
    ZYX = 5
    XYX = 6
    YZY = 7
    ZXZ = 8


class NiRotData(NiFloatData):
    euler_axis_order: int32 = AxisOrder.XYZ
    euler_data: tuple[NiFloatData, ...] = ()

    # provide access to related enums
    AxisOrder = AxisOrder

    def load(self, stream):
        num_keys = stream.read_uint()
        if num_keys:
            self.key_type = KeyType(stream.read_int())
            if self.key_type == KeyType.EULER_KEY:
                self.euler_axis_order = AxisOrder(stream.read_int())
                self.euler_data = (stream.read_type(NiFloatData),
                                   stream.read_type(NiFloatData),
                                   stream.read_type(NiFloatData))
            else:
                self.keys = stream.read_floats(num_keys, self.key_size)

    def save(self, stream):
        num_keys = self._num_keys()
        stream.write_uint(num_keys)
        if num_keys:
            stream.write_int(self.key_type)
            if self.key_type == KeyType.EULER_KEY:
                stream.write_int(self.euler_axis_order)
                self.euler_data[0].save(stream)
                self.euler_data[1].save(stream)
                self.euler_data[2].save(stream)
            else:
                stream.write_floats(self.keys)

    @property
    def values(self) -> ndarray:
        return self.keys[:, 1:5]

    @property
    def in_tans(self) -> ndarray:
        raise IndexError

    @property
    def out_tans(self) -> ndarray:
        raise IndexError

    @property
    def tcb(self) -> ndarray:
        return self.keys[:, -3:]

    @property
    def key_size(self) -> int:
        if self.key_type == KeyType.LIN_KEY:
            return 5  # (time, w, x, y, z)
        if self.key_type == KeyType.BEZ_KEY:
            return 5  # (time, w, x, y, z)
        if self.key_type == KeyType.TCB_KEY:
            return 8  # (time, w, x, y, z, tension, continuity, bias)
        raise Exception(f"{self.type} does not support '{self.key_type}'")

    def _num_keys(self):
        if self.key_type == KeyType.EULER_KEY:
            return any(len(e.keys) for e in self.euler_data)
        return len(self.keys)

    def convert_to_quaternions(self):
        if self.euler_data == ():
            return  # already using quaternions

        # TODO: support alternative axis orders
        assert self.euler_axis_order == AxisOrder.XYZ

        # extract keys and clear euler settings
        e_keys = [e.keys for e in self.euler_data]
        del self.key_type, self.euler_data, self.euler_axis_order

        if all(len(keys) == 0 for keys in e_keys):
            return  # no keys exist on any axis

        # The engine evaluates the three axis channels as independent curves
        # and composes them as R = Rz @ Ry @ Rx; the channels rarely share
        # timestamps. Sample every channel at the union of all key times
        # (linear interpolation, clamped at the ends) and compose per-time.
        # Composing only the axes that own a key at a given time (as done
        # previously) leaves the other axes at identity and scrambles the
        # animation.
        times = np.unique(np.concatenate([keys[:, 0] for keys in e_keys if len(keys)]))

        quats = None
        for i, keys in enumerate(e_keys):
            if len(keys) == 0:
                continue
            angles = np.interp(times, keys[:, 0], keys[:, 1])
            q = np.atleast_2d(quaternion_from_euler_angle(angles, i))
            quats = q if quats is None else quaternion_mul(q, quats)

        # keep successive quaternions on the same hemisphere so Blender's
        # componentwise key interpolation stays shortest-arc
        dots = np.einsum("ij,ij->i", quats[:-1], quats[1:])
        flips = np.cumprod(np.where(dots < 0, -1.0, 1.0))
        quats[1:] *= flips[:, None]

        self.keys = zeros(len(times), 5)
        self.keys[:, 0] = times
        self.keys[:, 1:5] = quats

    def apply_time_scale(self, scale: float):
        super().apply_time_scale(scale)
        for euler_data in self.euler_data:
            euler_data.apply_time_scale(scale)


if __name__ == "__main__":
    from es3.utils.typing import *
