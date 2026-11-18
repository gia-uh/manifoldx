"""The interactive frame's dt: the wall-clock interval between frames, smoothed.

Under vsync the screen shows each image for a whole number of refreshes, while a frame's start time wanders
with its own cost and with the swapchain's queue. Moving things by the raw start-to-start interval then
advances them by uneven steps that stay on screen for unrelated times, and motion judders. wgpu gives no
presentation timestamps, so the engine gives systems a running mean of the recent intervals instead, nudged
so the sum of the dts it hands out tracks the wall clock. An interval far over the mean (a load hitch) passes
through as it is: the time really went by, and spreading it over later frames would make them run fast.
"""

ALPHA = 0.05  # weight of the newest interval in the running mean: about the last twenty frames
GAIN = 0.05  # share of the gap between the wall clock and the dts handed out that one frame makes up
HITCH = 4.0  # an interval over this many times the mean passes through unsmoothed


class SmoothClock:
    """Turns each raw frame interval into the dt the frame's systems see."""

    def __init__(self, first: float = 1 / 60):
        self.mean = first
        self.behind = 0.0  # wall time not yet handed out as dt (negative: handed out ahead of the wall)

    def step(self, interval: float) -> float:
        """The dt for a frame that started `interval` seconds after the last one."""
        if interval > HITCH * self.mean:
            self.behind = 0.0
            return interval
        self.mean += ALPHA * (interval - self.mean)
        self.behind += interval
        dt = min(2 * self.mean, max(0.5 * self.mean, self.mean + GAIN * (self.behind - self.mean)))
        self.behind -= dt
        return dt
