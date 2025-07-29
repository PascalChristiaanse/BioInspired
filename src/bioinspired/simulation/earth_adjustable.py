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


class EarthSimulatorAdjustable(SimulationBase):
    """Earth Universe Simulator class.

    This class provides a simulation environment with only the Earth as a gravitational body.
    It inherits from the base simulator class and implements the required methods.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        coefficient_set: integrator.CoefficientSets = integrator.CoefficientSets.rk_4,
        integrator_type: str = "runge_kutta",
    ):
        self._stepsize = stepsize
        self._coefficient_set = coefficient_set
        self._integrator_type = integrator_type
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
            if self._integrator_type == "runge_kutta":
                self._integrator = integrator.runge_kutta_fixed_step(
                    self._stepsize,
                    coefficient_set=self._coefficient_set,
                )
            elif self._integrator_type == "bulirsch_stoer":
                self._integrator = integrator.bulirsch_stoer_fixed_step(
                    self._stepsize,
                    extrapolation_sequence=integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
                    maximum_number_of_steps=6,
                )
            elif self._integrator_type == "adams_bashforth_moulton":
                self._integrator = integrator.adams_bashforth_moulton_fixed_step(
                    self._stepsize,
                    relative_error_tolerance=1e-12,
                    absolute_error_tolerance=1e-12,
                    minimum_order=6,
                    maximum_order=11,
                )
            else:
                raise ValueError(f"Unknown integrator type: {self._integrator_type}")
        return self._integrator

    @override
    def _dump_integrator_settings(self) -> str:
        """Dump the integrator settings to a string representation in a JSON format."""
        return json.dumps({"step_size": 0.1, "type": "RK4"})
