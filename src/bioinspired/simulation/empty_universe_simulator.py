"""Empty Universe Simulation
This module provides an empty universe simulation class that inherits from the base simulator class.
It is used to create a simulation environment with no gravitational bodies, but with a spacecraft.
"""

import numpy as np
import json
from overrides import override

from tudatpy.interface import spice

from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import integrator
from tudatpy.numerical_simulation.environment_setup import (
    create_system_of_bodies,
    BodyListSettings,
)

from .simulation_base import SimulationBase


class EmptyUniverseSimulator(SimulationBase):
    """Empty Universe Simulator class.

    This class provides an empty universe simulation environment with no gravitational bodies.
    It inherits from the base simulator class and implements the required methods.
    """

    def __init__(
        self,
        tolerance: float = 1e-6,
        initial_timestep: float = 0.1,
        **kwargs,
    ):
        self._tolerance = tolerance
        self._initial_timestep = initial_timestep
        super().__init__(**kwargs)

    @override
    def _get_central_body(self) -> list[str]:
        return ["SSB"]

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
                minimum_step=1e-4, maximum_step=1
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

    @override
    def get_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        # Create an empty body model.
        if self._body_model is None:
            # Create settings for celestial bodies
            body_settings = BodyListSettings(
                self.global_frame_origin, self.global_frame_orientation
            )
            # # Create environment
            self._body_model = create_system_of_bodies(body_settings)
            # self._body_model.create_empty_body(""
        return self._body_model


def main():
    """Main function to test the empty universe simulation."""
    from bioinspired.spacecraft import SimpleCraft

    # Load spice kernels
    spice.load_standard_kernels()

    # Create an instance of the EmptyUniverseSimulator
    simulator = EmptyUniverseSimulator()

    spacecraft = SimpleCraft(
        simulation=simulator, initial_state=np.array([6378e3, 0, 0, 0, 8e3, 0])
    )
    print("Spacecraft name:", spacecraft.get_name())

    # Check all bodies in the system
    print("Bodies in the system:")
    for body in simulator.get_body_model().list_of_bodies():
        print(f" - {body}")

    # Check the spacecraft's acceleration settings
    acceleration_settings = spacecraft._get_acceleration_settings()
    print("Acceleration settings for spacecraft:", acceleration_settings)

    result = simulator.run(0, 100)
    print("Simulation completed successfully.")
    print(result.propagation_results.state_history)


if __name__ == "__main__":
    main()
