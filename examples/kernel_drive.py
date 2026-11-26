"""One Python drive kernel, scheduled after Python steering in a native World."""

from manifoldx import World, clamp, f32, kernel, vec3
from manifoldx.components import Component


class Car(Component):
    speed: f32
    target_speed: f32
    pos: vec3
    forward: vec3


class Player(Component):
    """Select the car for the Python input system."""


@kernel
def drive(car: Car, dt: f32) -> None:
    car.speed += clamp(car.target_speed - car.speed, -3.0 * dt, 2.0 * dt)
    car.pos += car.forward * car.speed * dt


def steering_target(keys):
    return 30.0 if "w" in keys else 0.0


def install_steering(world, keys_at):
    player = world.query(Player)
    target_speed = world.field(Car, "target_speed")
    previous = None

    @world.system(writes=(target_speed,))
    def steer(snapshot, commands):
        nonlocal previous
        target = steering_target(keys_at(snapshot.clock.frame))
        if target != previous:
            commands.set(player, target_speed, target)
            previous = target


def make_world(capacity):
    world = World(capacity)
    world.register(Car)
    world.register(Player)
    return world


def main():
    world = make_world(2)
    world.spawn(Car(forward=(1, 0, 0)), Player())
    world.spawn(Car(forward=(1, 0, 0)))
    install_steering(world, lambda frame: {"w"} if frame < 120 else set())
    world.kernel(drive, target="cpu")
    report = world.run(frames=180, paced=False)
    cars = world.query(Car).numpy()
    print(f"{report['kernel_backend']} · {report['kernel_dispatches']} dispatches")
    print(f"Player speed {cars[0, 0]:.3f}, position {cars[0, 2]:.3f}")
    print(f"Other car speed {cars[1, 0]:.3f}, position {cars[1, 2]:.3f}")


def desktop_scene():
    """Show both cars. Hold W to accelerate the red car and release to brake."""
    import manifoldx as mx
    from manifoldx.components import Material, Mesh, Transform

    engine = mx.Engine("Resident Drive", width=800, height=600, max_entities=32)
    world = engine.world
    world.register(Car)
    world.register(Player)
    engine.camera.set_pose(position=(1.5, 5, 10), target=(1.5, 0, 0))
    mesh = mx.geometry.cube(0.65, 0.65, 0.65)
    engine.spawn(
        Mesh(mesh),
        Material(mx.material.flat("#ff3010")),
        Transform(pos=(-200, 0, 0)),
        Car(pos=(-2, 0, 0), forward=(1, 0, 0)),
        Player(),
    )
    engine.spawn(
        Mesh(mesh),
        Material(mx.material.flat("#10e040")),
        Transform(pos=(200, 0, 0)),
        Car(pos=(-2, 0, -1.6), forward=(1, 0, 0)),
    )
    install_steering(world, lambda _: engine.input.pressed_keys)
    engine.bind_position(drive, field="pos", target="wgsl")
    return engine


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["--desktop"]:
        desktop_scene().run()
    else:
        main()
