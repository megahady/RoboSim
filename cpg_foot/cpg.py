"""A small Central Pattern Generator (CPG) for one leg.

Two coupled phase oscillators drive a Cartesian foot target in the leg's
chain plane (u = radial, v = up).  Their phase-locking is set so the *pair*
of outputs (x = cos, y = sin) sweeps the top half of an ellipse sitting on
the ground line:

    swing  (v > 0): the foot lifts and swings, tracing the upper half of
                     an ellipse centered on the ground line,
    stance (v = 0): the foot tracks back along the ground line.

The oscillator coupling uses the standard Kuramoto phase model with a
desired phase offset; amplitudes relax exponentially to the commanded
ellipse half-axes.
"""

from __future__ import annotations

import math


class Cpg:
    """Dual-oscillator CPG producing a half-elliptic foot path."""

    def __init__(self, freq_hz: float = 0.6, amp_u: float = 0.05,
                 amp_v: float = 0.06, coupling: float = 6.0,
                 phase_offset: float = math.pi / 2):
        self.w = 2.0 * math.pi * freq_hz
        self.amp_u = amp_u
        self.amp_v = amp_v
        self.coupling = coupling
        self.phase_offset = phase_offset
        self.phi = [0.0, -phase_offset]     # locked to lag by phase_offset
        self.x = [0.0, 0.0]

    def reset(self):
        self.phi = [0.0, -self.phase_offset]
        self.x = [0.0, 0.0]

    def step(self, dt: float):
        """Advance the oscillators one dt, return normalized (u, v) ellipses."""
        w = self.w
        g = self.coupling
        d = self.phase_offset
        dphi1 = w + g * math.sin(self.phi[1] - self.phi[0] - d)
        dphi2 = w + g * math.sin(self.phi[0] - self.phi[1] + d)
        self.phi[0] += dphi1 * dt
        self.phi[1] += dphi2 * dt
        # amplitude relaxation toward the commanded ellipse axes
        for i in range(2):
            target = self.amp_u if i == 0 else self.amp_v
            self.x[i] += (target - self.x[i]) * min(1.0, dt * 4.0)
        return self.x[0] * math.cos(self.phi[0]), self.x[1] * math.sin(self.phi[1])

    def outp(self) -> tuple[float, float]:
        """Quasi-static ellipse outputs (no integration side effects)."""
        return (self.x[0] * math.cos(self.phi[0]),
                self.x[1] * math.sin(self.phi[1]))