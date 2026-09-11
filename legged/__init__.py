"""Legged-robot control simulation toolkit (systematic formulation).

The 6-step methodology from the repository notes, codified as a reusable
package:

    Step 1-2 : model + env            -> legged.env.LeggedEnv (floating base,
                                         sensor wrapper, reset/step)
    Step 3   : the controller         -> legged.gait (layered: reference,
                                         PD + bias feed-forward)
    Step 4   : control/physics rates  -> the env's step() (mj_step1/2 split)
    Step 5   : metrics                -> legged.metrics (efficacy numbers)
    Step 6   : experiments + demo     -> legged.evaluate, legged.run_experiment

Engineering decision (the "systematic" part): every quantity the controller
can see, and every number used to judge it, is *defined* in one place so that
"efficacy" is an unambiguous, repeatable number — not a videoclip.
"""

from legged.env import LeggedEnv
from legged.gait import LayeredController, PhaseGait, JointPD
from legged.metrics import Metrics

__all__ = ["LeggedEnv", "LayeredController", "PhaseGait", "JointPD", "Metrics"]