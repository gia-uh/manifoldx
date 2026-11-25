//! Static segments and vehicle footprints, resolved with rigid-body impulses.

use super::sim::{add, cross, dot, multiply, scale, Body, Sim, V3};
use super::terrain::Obstacle;

const SLOP: f64 = 0.01;
const PUSH: f64 = 0.8;
const RESTING: f64 = 1.0;
const RESTITUTION: f64 = 0.2;

type V2 = [f64; 2];

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub a: usize,
    pub b: Option<usize>,
    pub normal: V3,
    pub impulse: V3,
}

fn project(a: V2, b: V2) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn offset(a: V2, b: V2) -> V2 {
    [a[0] - b[0], a[1] - b[1]]
}

fn perpendicular([x, z]: V2) -> V2 {
    [-z, x]
}

struct Footprint {
    center: V2,
    center_y: f64,
    y: [f64; 2],
    axes: [V2; 2],
    half: V2,
}

impl Footprint {
    fn of(sim: &Sim) -> Self {
        let [x, y, z] = sim.pose().0;
        let (axes, vertical) = sim.body_bounds();
        let [length, _, width] = sim.spec().size;
        let [forward, up, side] = axes;
        let yaw = forward[0].hypot(forward[2]);
        let heading = if yaw > 1e-9 {
            [forward[0] / yaw, forward[2] / yaw]
        } else {
            let length = side[0].hypot(side[2]);
            [side[2] / length, -side[0] / length]
        };
        let vertical_radius = (forward[1].abs() * length + side[1].abs() * width) * 0.5;
        let [lower, upper] = vertical.map(|offset| y + up[1] * offset);
        Self {
            center: [x, z],
            center_y: y,
            y: [
                lower.min(upper) - vertical_radius,
                lower.max(upper) + vertical_radius,
            ],
            axes: [heading, perpendicular(heading)],
            half: [length * 0.5, width * 0.5],
        }
    }

    fn radius(&self, axis: V2) -> f64 {
        self.half[0] * project(self.axes[0], axis).abs()
            + self.half[1] * project(self.axes[1], axis).abs()
    }

    fn span(&self, axis: V2) -> [f64; 2] {
        let center = project(self.center, axis);
        let radius = self.radius(axis);
        [center - radius, center + radius]
    }

    fn bounds(&self) -> (V2, V2) {
        let [xmin, xmax] = self.span([1.0, 0.0]);
        let [zmin, zmax] = self.span([0.0, 1.0]);
        ([xmin, zmin], [xmax, zmax])
    }

    fn against(
        &self,
        axes: impl IntoIterator<Item = V2>,
        other_span: impl Fn(V2) -> [f64; 2],
        other_y: [f64; 2],
    ) -> Option<Hit> {
        let bottom = self.y[0].max(other_y[0]);
        let top = self.y[1].min(other_y[1]);
        if bottom >= top {
            return None;
        }
        let mut closest = f64::INFINITY;
        let mut normal = [0.0; 2];
        for axis in axes {
            let [a0, a1] = self.span(axis);
            let [b0, b1] = other_span(axis);
            let left = a1 - b0;
            let right = b1 - a0;
            let depth = left.min(right);
            if depth <= 0.0 {
                return None;
            }
            if depth < closest {
                closest = depth;
                normal = if left <= right {
                    [-axis[0], -axis[1]]
                } else {
                    axis
                };
            }
        }
        let tangent = perpendicular(normal);
        let [a0, a1] = self.span(tangent);
        let [b0, b1] = other_span(tangent);
        let along = (a0.max(b0) + a1.min(b1)) * 0.5;
        let across = (self.span(normal)[0] + other_span(normal)[1]) * 0.5;
        // Use the body's centre vertically when the obstacle spans it. Full
        // height centred impacts then have no artificial torque.
        let y = self.center_y.clamp(bottom, top);
        Some(Hit {
            depth: closest,
            normal: [normal[0], 0.0, normal[1]],
            point: [
                normal[0] * across + tangent[0] * along,
                y,
                normal[1] * across + tangent[1] * along,
            ],
        })
    }

    fn wall(&self, wall: Obstacle) -> Option<Hit> {
        if wall.strength == 0.0 {
            return None;
        }
        let [a, b] = wall.ends;
        let [dx, dz] = offset(b, a);
        let length = dx.hypot(dz);
        self.against(
            [self.axes[0], self.axes[1], [-dz / length, dx / length]],
            |axis| {
                let a = project(a, axis);
                let b = project(b, axis);
                [a.min(b), a.max(b)]
            },
            wall.y,
        )
    }
}

struct Hit {
    depth: f64,
    normal: V3,
    point: V3,
}

fn lever(body: &Body, point: V3) -> V3 {
    add(point, scale(body.pos, -1.0))
}

fn point_velocity(body: &Body, r: V3) -> V3 {
    add(body.vel, cross(body.ang, r))
}

fn inverse_mass(body: &Body, r: V3, axis: V3) -> f64 {
    let torque = cross(r, axis);
    body.inv_mass + dot(torque, multiply(body.inv, torque))
}

fn apply_impulse(body: &mut Body, r: V3, impulse: V3) {
    body.vel = add(body.vel, scale(impulse, body.inv_mass));
    body.ang = add(body.ang, multiply(body.inv, cross(r, impulse)));
    body.impulse += dot(impulse, impulse).sqrt();
}

fn resolve(
    a: &mut Body,
    mut b: Option<&mut Body>,
    hit: Hit,
    strength: f64,
    friction: f64,
) -> Option<(V3, V3)> {
    let normal = hit.normal;
    let ra = lever(a, hit.point);
    let rb = b.as_ref().map_or([0.0; 3], |b| lever(b, hit.point));
    let relative = add(
        point_velocity(a, ra),
        b.as_ref()
            .map_or([0.0; 3], |b| scale(point_velocity(b, rb), -1.0)),
    );
    let speed = dot(relative, normal);
    let inv_mass_sum = a.inv_mass + b.as_ref().map_or(0.0, |b| b.inv_mass);
    let mut applied = None;
    if speed < 0.0 {
        let mass_at = |axis| {
            inverse_mass(a, ra, axis) + b.as_ref().map_or(0.0, |b| inverse_mass(b, rb, axis))
        };
        let bounce = if speed < -RESTING { RESTITUTION } else { 0.0 };
        let jn = -(1.0 + bounce) * speed / mass_at(normal);
        let tangent_velocity = add(relative, scale(normal, -speed));
        let tangential = dot(tangent_velocity, tangent_velocity).sqrt();
        let tangent = scale(tangent_velocity, 1.0 / tangential.max(1e-9));
        let jt = if tangential > 1e-9 {
            (tangential / mass_at(tangent)).min(friction * jn)
        } else {
            0.0
        };
        let impulse = scale(add(scale(normal, jn), scale(tangent, -jt)), strength);
        apply_impulse(a, ra, impulse);
        if let Some(b) = &mut b {
            apply_impulse(b, rb, scale(impulse, -1.0));
        }
        applied = Some((normal, impulse));
    }
    let push = (hit.depth - SLOP).max(0.0) * PUSH * strength / inv_mass_sum;
    a.pos = add(a.pos, scale(normal, push * a.inv_mass));
    if let Some(b) = b {
        b.pos = add(b.pos, scale(normal, -push * b.inv_mass));
    }
    applied
}

pub(super) fn walls(
    sim: &mut Sim,
    ground: &super::Ground,
    index: usize,
    friction: f64,
    contacts: &mut Vec<Contact>,
) {
    let (lo, hi) = Footprint::of(sim).bounds();
    for wall in ground.obstacles(lo, hi) {
        let Some(hit) = Footprint::of(sim).wall(wall) else {
            continue;
        };
        let mut body = sim.body();
        if let Some((normal, impulse)) = resolve(&mut body, None, hit, wall.strength, friction) {
            contacts.push(Contact {
                a: index,
                b: None,
                normal,
                impulse,
            });
        }
        sim.set_body(body);
    }
}

pub(super) fn pairs(sims: &mut [Sim], friction: f64, contacts: &mut Vec<Contact>) {
    let mut by_x: Vec<_> = sims
        .iter()
        .enumerate()
        .map(|(i, sim)| {
            let (low, high) = Footprint::of(sim).bounds();
            (low[0], high[0], i)
        })
        .collect();
    by_x.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.2.cmp(&b.2)));
    let mut candidates = Vec::new();
    for (slot, &(_, high, i)) in by_x.iter().enumerate() {
        for &(low, _, j) in &by_x[slot + 1..] {
            if low >= high {
                break;
            }
            candidates.push((i.min(j), i.max(j)));
        }
    }
    candidates.sort_unstable();
    for (i, j) in candidates {
        let (before, after) = sims.split_at_mut(j);
        let (a, b) = (&mut before[i], &mut after[0]);
        let (pa, pb) = (Footprint::of(a), Footprint::of(b));
        let Some(hit) = pa.against(pa.axes.into_iter().chain(pb.axes), |u| pb.span(u), pb.y) else {
            continue;
        };
        let (mut ba, mut bb) = (a.body(), b.body());
        if let Some((normal, impulse)) = resolve(&mut ba, Some(&mut bb), hit, 1.0, friction) {
            contacts.push(Contact {
                a: i,
                b: Some(j),
                normal,
                impulse,
            });
        }
        a.set_body(ba);
        b.set_body(bb);
    }
}
