"""The native renderer accepts the camera's public three-value sequences."""

import numpy as np
import pytest

_camera_input = pytest.importorskip("manifoldx._desktop")._camera_input


def test_camera_input_normalises_tuples_lists_and_arrays():
    from manifoldx.camera import Camera

    camera = Camera()
    camera.position = (1.0, 2.0, 3.0)
    camera.target = [0.0, 0.0, 0.0]
    camera.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    position, target, up, fov, near, far = _camera_input(camera)

    assert position == [1.0, 2.0, 3.0]
    assert target == [0.0, 0.0, 0.0]
    assert up == [0.0, 1.0, 0.0]
    assert (fov, near, far) == (60, 0.1, 1000.0)


def test_camera_input_keeps_numpy_values():
    from manifoldx.camera import Camera

    camera = Camera(position=np.array([1.5, 2.5, 3.5], dtype=np.float32))

    position, *_ = _camera_input(camera)

    assert position == [1.5, 2.5, 3.5]
