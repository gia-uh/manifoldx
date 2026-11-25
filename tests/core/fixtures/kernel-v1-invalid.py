"""Source-only negative fixture for the kernel import-time diagnostic."""

from manifoldx import Car, f32, kernel


@kernel
def invalid_import(car: Car, dt: f32) -> None:
    import numpy
