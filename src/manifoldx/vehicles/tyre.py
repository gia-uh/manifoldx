"""The tyre: Pacejka's magic formula without its peak factor, combined over both slips, with relaxation.

Slips are kept as states that relax toward their steady values over RELAX metres rolled: at speed they are
the usual slip ratio and tangent of the slip angle; at rest they become the tread's deflection over RELAX, so a
parked car stands on a spring instead of creeping.
"""

from __future__ import annotations

import numpy as np

RELAX = 0.3  # metres of rolling over which a slip settles
V_MIN = 3.0  # m/s under which the tyre adds damping that fades in as the car slows
LOW_SPEED_DAMPING = 0.5  # share of critical damping that damping reaches at rest


def magic(s, b, c, e):
    """sin(c·atan(b·s − e·(b·s − atan(b·s)))): the share of μ·load a tyre gives at slip s, odd in s."""
    bs = b * np.asarray(s, float)
    return np.sin(c * np.arctan(bs - e * (bs - np.arctan(bs))))


def peak_slip(b, c, e):
    """The slip where magic peaks, found on a grid fine enough for the assists."""
    s = np.linspace(0.0, 1.0, 2001)
    return float(s[np.argmax(magic(s, b, c, e))])


def forces(kappa, alpha, load, mu, tyre, kappa_peak, alpha_peak):
    """Longitudinal and lateral force of each tyre from its slip ratio and slip-angle tangent.

    Both slips are measured in their own peaks, combined into one length, and each direction takes its share
    of its own curve at that length: the force never passes mu·load, and a locked wheel slides along its slide.
    tyre is (..., 7) in TYRE_FIELDS order. Fy pushes against alpha.
    """
    kn, an = kappa / kappa_peak, alpha / alpha_peak
    rho = np.hypot(kn, an)
    safe = np.where(rho > 1e-12, rho, 1.0)
    fx = magic(safe * kappa_peak, tyre[..., 4], tyre[..., 5], tyre[..., 6]) * kn / safe
    fy = -magic(safe * alpha_peak, tyre[..., 1], tyre[..., 2], tyre[..., 3]) * an / safe
    scale = np.where(rho > 1e-12, mu * load, 0.0)
    return fx * scale, fy * scale


def slope(s, b, c, e, h=1e-4):
    """d magic / ds at s."""
    return (magic(s + h, b, c, e) - magic(s - h, b, c, e)) / (2 * h)
