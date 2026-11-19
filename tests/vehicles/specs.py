"""Vehicle specs the vehicle tests share: a 1 200 kg car whose CG sits 1.15 m behind the front axle."""

from manifoldx.vehicles import Engine, Gearbox, Tyre, VehicleSpec, Wheel, box_inertia

A, B = 1.15, 1.45  # metres from the CG to the front and the rear axle
TRACK = 1.5
RADIUS = 0.31


def car(mass=1200.0, drive="rear", tyre=Tyre(), cd=0.35, rolling=0.015, n_wheels=4, power_kw=85.0):
    """A test car. drive is "rear", "front" or "all"; n_wheels 6 doubles the rear axle."""
    size = (4.4, 1.45, 1.75)
    corner = mass * 9.81 / 4
    k = corner / 9.81 * (2 * 3.14159 * 1.4) ** 2  # 1.4 Hz per corner
    c = 2 * 0.35 * (k * corner / 9.81) ** 0.5
    def wheel(x, z, front):
        return Wheel(mount=(x, -0.15, z), radius=RADIUS, travel=0.2, spring=k, damper=c, tyre=tyre,
                     driven=(drive == "all") or (drive == "front") == front, steered=front,
                     brake_share=0.3 if front else 0.2, handbrake=not front)
    wheels = [wheel(A, -TRACK / 2, True), wheel(A, TRACK / 2, True), wheel(-B, -TRACK / 2, False),
              wheel(-B, TRACK / 2, False)]
    if n_wheels == 6:
        wheels = wheels[:2] + [wheel(-B, -TRACK / 2, False), wheel(-B, TRACK / 2, False),
                               wheel(-B - 1.2, -TRACK / 2, False), wheel(-B - 1.2, TRACK / 2, False)]
        share = 0.2 * 2 / 4
        wheels = wheels[:2] + [Wheel(**{**w.__dict__, "brake_share": share}) for w in wheels[2:]]
    peak = power_kw * 1000 / (5500 * 2 * 3.14159 / 60)  # torque at the power peak
    return VehicleSpec(mass=mass, inertia=box_inertia(mass, size), size=size, wheels=tuple(wheels),
                       engine=Engine(idle_rpm=850, peak_torque_rpm=3500, peak_power_rpm=5500,
                                     idle_torque=0.6 * peak, peak_torque=1.3 * peak, power_torque=peak),
                       gearbox=Gearbox(ratios=(3.6, 2.1, 1.4, 1.0, 0.8), final=3.9),
                       arb=(9000.0, 4000.0), cd=cd, area=2.0, rolling=rolling, max_steer=0.6,
                       brake_torque=5000.0)
