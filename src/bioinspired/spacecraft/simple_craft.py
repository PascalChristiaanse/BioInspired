"""Simple spacecraft class for bioinspired design.
This module contains a simple spacecraft class that inherits from the base spacecraft class.
It provides a basic implementation of a spacecraft design with a name and a method to get its model.
"""

import numpy as np

from tudatpy.numerical_simulation.propagation_setup import propagator

from .spacecraft_base import SpacecraftBase


class SimpleCraft(SpacecraftBase):
    """Simple spacecraft class.

    This class provides a basic implementation of a spacecraft design with a name and a method to get its model.
    It inherits from the base spacecraft class and implements the required methods.
    """

    def __init__(self, initial_state: np.ndarray, simulation):
        """Initialize the simple spacecraft with a name."""
        super().__init__("SimpleCraft", simulation, initial_state=initial_state)
        self._mass = 1000.0  # Spacecraft mass [kg]

    def _get_acceleration_settings(self):
        return super()._get_acceleration_settings()

    def _get_propagator(self) -> propagator.PropagatorSettings:
        """Return the propagator settings for the spacecraft."""
        # Create a propagator settings object for the spacecraft
        # Create propagation settings.
        return propagator.translational(
            self._simulation._get_central_body(),
            self._get_acceleration_model(),
            [self.name],
            self._initial_state,
            self._simulation._start_epoch,
            self._simulation._get_integrator(),
            self._get_termination(),
            # output_variables=dependent_variables_to_save
        )


def main():
    """Main function to demonstrate the SimpleCraft class."""
    simple_craft = SimpleCraft()
    print("SimpleCraft instance created with name:", simple_craft.get_name())


if __name__ == "__main__":
    main()
