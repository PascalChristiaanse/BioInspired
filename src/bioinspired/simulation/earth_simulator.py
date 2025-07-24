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

    def __init__(
        self,
        tolerance: float = 1e-6,
        initial_timestep: float = 0.1,
    ):
        self._tolerance = tolerance
        self._initial_timestep = initial_timestep
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
            block_indices = [
                (0, 0, 3, 1),
                (3, 0, 3, 1),
                (6, 0, 4, 1),
                (10, 0, 3, 1),
            ]
            step_size_control = integrator.step_size_control_blockwise_scalar_tolerance(
                block_indices, self._tolerance, self._tolerance
            )

            # Create step size validation settings
            step_size_validation = integrator.step_size_validation(
                minimum_step=1e-4, maximum_step=10000
            )

            self._integrator = integrator.bulirsch_stoer_variable_step(
                initial_time_step=self._initial_timestep,
                extrapolation_sequence=integrator.ExtrapolationMethodStepSequences.deufelhard_sequence,
                maximum_number_of_steps=4,
                step_size_control_settings=step_size_control,
                step_size_validation_settings=step_size_validation,
                assess_termination_on_minor_steps=False,
            )
        return self._integrator

    @override
    def _dump_integrator_settings(self) -> str:
        """Dump the integrator settings to a string representation in a JSON format."""

        return json.dumps(
            {
                "block_indices": [
                    (0, 0, 3, 1),
                    (3, 0, 3, 1),
                    (6, 0, 4, 1),
                    (10, 0, 3, 1),
                ],
                "step_size_control": "step_size_control_blockwise_scalar_tolerance",
                "step_size_validation": {
                    "minimum_step": 1e-4,
                    "maximum_step": 10000,
                },
                "integrator_type": "bulirsch_stoer_variable_step",
                "initial_time_step": self._initial_timestep,
                "extrapolation_sequence": "deufelhard_sequence",
                "maximum_number_of_steps": 4,
                "assert_termination_on_minor_steps": False,
                "tolerance": self._tolerance,
            }
        )
