"""Empty Universe Simulation
This module provides an empty universe simulation class that inherits from the base simulator class.
It is used to create a simulation environment with no gravitational bodies, but with a spacecraft.
"""

import numpy as np

from tudatpy.interface import spice
from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.environment_setup import (
    create_system_of_bodies,
)
from tudatpy.numerical_simulation.environment_setup import BodyListSettings

from tudatpy.numerical_simulation.propagation_setup import integrator
from tudatpy.numerical_simulation.propagation_setup.integrator import (
    IntegratorSettings,
)

try:
    from .simulation_base import SimulatorBase
except ImportError:
    from simulation_base import SimulatorBase


class EmptyUniverseSimulator(SimulatorBase):
    """Empty Universe Simulator class.

    This class provides an empty universe simulation environment with no gravitational bodies.
    It inherits from the base simulator class and implements the required methods.
    """

    def __init__(self):
        super().__init__()
        self._body_model: SystemOfBodies = self._get_body_model()
        self._integrator: IntegratorSettings = self._get_integrator()

    def _get_integrator(self):
        """Return the integrator settings object."""
        # Create numerical integrator settings.
        if self._integrator is None:
            fixed_step_size = 10.0
            self._integrator = integrator.runge_kutta_fixed_step(
                fixed_step_size,
                coefficient_set=integrator.CoefficientSets.rk_4,
            )
        return self._integrator

    def _get_body_model(self):
        """Return the body model object."""
        # Create an empty body model.
        if self._body_model is None:
            # Create settings for celestial bodies
            global_frame_origin = "SSB"
            global_frame_orientation = "ECLIPJ2000"
            body_settings = BodyListSettings(
                global_frame_origin, global_frame_orientation
            )
            # Create environment
            self._body_model = create_system_of_bodies(body_settings)
        return self._body_model

    def _get_propagator(self):
        """Return the propagator settings object."""
        raise NotImplementedError(
            "Propagator settings are not defined for an empty universe."
        )


def main():
    """Main function to test the empty universe simulation."""
    # Load spice kernels
    spice.load_standard_kernels()

    # Create an instance of the EmptyUniverseSimulator
    simulator = EmptyUniverseSimulator()

    bodies = simulator._get_body_model()
    print("Bodies in the simulation:")
    print(bodies)


if __name__ == "__main__":
    main()
