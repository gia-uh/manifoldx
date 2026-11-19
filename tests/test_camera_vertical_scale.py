import numpy as np
import pytest

from manifoldx.camera import Camera


def _camera(k):
    cam = Camera(position=(0.0, 50.0, 80.0), target=(0.0, 0.0, 0.0), fov=60, near=0.1, far=1000.0)
    cam.vertical_scale = k
    return cam


def test_a_stretched_camera_shows_heights_times_its_scale():
    flat, tall = _camera(1.0), _camera(3.0)
    xy_tall, d_tall = tall.project([(5.0, 4.0, -10.0)], 800, 600)
    xy_flat, d_flat = flat.project([(5.0, 12.0, -10.0)], 800, 600)
    assert np.allclose(xy_tall, xy_flat) and np.allclose(d_tall, d_flat)


def test_sea_level_does_not_move():
    flat, tall = _camera(1.0), _camera(3.0)
    p = [(7.0, 0.0, -20.0)]
    assert np.allclose(tall.project(p, 800, 600)[0], flat.project(p, 800, 600)[0])


def test_right_and_up_stay_unit_and_unscaled():
    flat, tall = _camera(1.0), _camera(5.0)
    assert np.allclose(tall.get_right(), flat.get_right())
    assert np.allclose(tall.get_up(), flat.get_up())
    assert np.linalg.norm(tall.get_up()) == pytest.approx(1.0, abs=1e-6)
