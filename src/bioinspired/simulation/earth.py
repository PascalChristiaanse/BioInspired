"""Earth only universe simulator.

This module provides an empty universe simulation class that inherits from the base simulator class.
It is used to create a simulation environment with only the Earth as a gravitational body, and a spacecraft.
"""

import numpy as np
import json
from overrides import override

from tudatpy.interface import spice
spice.load_standard_kernels()

from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import integrator
from tudatpy.numerical_simulation.environment_setup import (
    create_system_of_bodies,
    BodyListSettings,
    get_default_body_settings,
)

from .simulation_base import SimulationBase
from bioinspired.spacecraft import SimpleCraft


class EarthSimulator(SimulationBase):
    """Earth Universe Simulator class.

    This class provides a simulation environment with only the Earth as a gravitational body.
    It inherits from the base simulator class and implements the required methods.
    """

    @override
    def __init__(self):
        super().__init__()
        

    @override
    def _get_central_body(self) -> list[str]:
        return ["Earth"]

    @override
    def get_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        if self._body_model is None:
            
            # Create settings for celestial bodies
            body_settings = get_default_body_settings(
                ["Earth"], self.global_frame_origin, self.global_frame_orientation
            )

            # # Create environment
            self._body_model = create_system_of_bodies(body_settings)

        return self._body_model

    @override
    def _get_integrator(self) -> integrator.IntegratorSettings:
        """Return the integrator settings object."""
        # Create numerical integrator settings.
        if self._integrator is None:
            fixed_step_size = 10.0
            self._integrator = integrator.runge_kutta_fixed_step(
                fixed_step_size,
                coefficient_set=integrator.CoefficientSets.rk_4,
            )
        return self._integrator

    @override
    def _dump_integrator_settings(self) -> str:
        """Dump the integrator settings to a string representation in a JSON format."""
        return json.dumps({"step_size": 100.0, "type": "RK4"})
