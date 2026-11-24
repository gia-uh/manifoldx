"""Source-only AST fixtures. The compiler tests in E2 consume these spans."""

from manifoldx import Car, clamp, f32, kernel


@kernel
def drive(car: Car, dt: f32) -> None:
    car.speed += clamp(car.target_speed - car.speed, -3.0 * dt, 2.0 * dt)
    car.pos += car.forward * car.speed * dt
