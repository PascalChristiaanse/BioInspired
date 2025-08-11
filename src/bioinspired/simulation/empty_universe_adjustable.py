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



class EmptyUniverseSimulatorAdjustable(SimulationBase):
    """Empty Universe Simulator class.

    This class provides an empty universe simulation environment with no gravitational bodies.
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
        return ["SSB"]

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
                    maximum_number_of_steps=6
                )
            elif self._integrator_type == "adams_bashforth_moulton":
                self._integrator = integrator.adams_bashforth_moulton_fixed_step(
                    self._stepsize,
                    relative_error_tolerance=1e-12,
                    absolute_error_tolerance=1e-12,
                    minimum_order=6,
                    maximum_order=11
                )
            else:
                raise ValueError(f"Unknown integrator type: {self._integrator_type}")
        return self._integrator

    @override
    def _dump_integrator_settings(self) -> str:
        """Dump the integrator settings to a string representation in a JSON format."""
        integrator_info = {
            "step_size": self._stepsize,
            "integrator_type": self._integrator_type,
        }
        
        if self._integrator_type in ["runge_kutta", "adams_bashforth_moulton"]:
            coefficient_name = self._coefficient_set.name if hasattr(self._coefficient_set, 'name') else str(self._coefficient_set)
            integrator_info["coefficient_set"] = coefficient_name
        
        return json.dumps(integrator_info)

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
    simulator = EmptyUniverseSimulatorAdjustable()

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
