"""Lander 2 spacecraft module.
This module defines the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
It is based on the paper
"""

import numpy as np

from overrides import override
from numpy.linalg import norm

from tudatpy.numerical_simulation.propagation_setup import (
    acceleration,
    torque,
)

from bioinspired.simulation.simulation_base import SimulationBase

# from bioinspired.simulation import EmptyUniverseSimulator

from .rotating_spacecraft_base import RotatingSpacecraftBase
from .JSON_spacecraft_base import JSONSpacecraftBase

from bioinspired.controllers import ControllerBase


class Lander2(RotatingSpacecraftBase, JSONSpacecraftBase):
    """Lander 2 spacecraft design.

    This class implements the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
    It provides specific configurations and methods for the Lander 2 spacecraft.
    """

    def __init__(
        self,
        simulation: SimulationBase,
        controller: ControllerBase,
        initial_state: np.ndarray,
    ):
        """Initialize the Lander 2 spacecraft with a name and initial state."""

        super().__init__(
            initial_rotational_state=initial_state[6:],
            name="Lander 2", 
            simulation=simulation, 
            initial_state=initial_state[:6]
        )
        self.controller: ControllerBase = controller
        self._torque_settings = {}

        # self._simulation._get_body_model().get(self.name).constant_mass = self._physical_properties["dry_mass"] + self._physical_properties["fuel_mass"]
        self._simulation.get_body_model().get(self.name).mass = 10
        # self.setup_engine_model()

    @override
    def required_properties(self) -> dict[str, list[str]]:
        """Return a list of required properties for the spacecraft configuration."""
        return {
            "engines": ["location", "thrust_direction", "maximum_thrust"],
            "fuel_mass": [],
            "dry_mass": [],
            "inertia_tensor": [],
            "center_of_mass": [],
        }

    @property
    @override
    def mass(self) -> float:
        """Return the mass of the Lander 2 spacecraft."""
        return self._fuel_mass + self._dry_mass  # pylint: disable=no-member

    @property
    @override
    def inertia_tensor(self) -> np.ndarray[np.ndarray]:
        """Return the inertia tensor of the Lander 2 spacecraft."""
        return np.array(self._inertia_tensor)  # pylint: disable=no-member

    @property
    @override
    def center_of_mass(self) -> np.ndarray:
        """Return the center of mass of the Lander 2 spacecraft."""
        return np.array(self._center_of_mass)  # pylint: disable=no-member

    @override
    def _get_torque_settings(
        self,
    ) -> dict[str, dict[str, list[torque.TorqueSettings]]]:
        """Compiles the torque model for the spacecraft."""
        self._torque_settings = {
            # self.name: {self.name: [torque.custom_torque(self.get_torque_vector)]}
        }
        return self._torque_settings

    def get_thrust_vector(self, current_time: float) -> np.ndarray:
        """Calculate the thrust based on the control vector.
        Computed through sum of the forces due to RCS thrusters."""
        control_vector = self.controller.get_control_action()
        total_thrust = np.zeros(3, 1)
        for i, direction, max_thrust in enumerate(
            zip(self._engine_directions, self._engine_max_thrusts)  # pylint: disable=no-member
        ):
            total_thrust += (
                (direction / norm(direction)) * control_vector[i] * max_thrust
            )
        return total_thrust

    def get_thrust_direction(self, current_time: float) -> np.ndarray:
        """Calculate the thrust direction based on the control vector."""
        if current_time == current_time:
            thrust_vector = self.get_thrust_vector(current_time)
            if norm(thrust_vector) == 0:
                return np.zeros(3)
            return thrust_vector / norm(thrust_vector)
        # If no computation is to be done, return zeros
        else:
            return np.zeros([3, 1])

    def get_thrust_magnitude(self, current_time: float) -> float:
        """Calculate the total thrust magnitude based on the control vector."""
        if current_time == current_time:
            thrust_vector = self.get_thrust_vector(current_time)
            return norm(thrust_vector)
        # If no computation is to be done, return zeros
        else:
            return 0.0

    def get_torque_vector(self, current_time: float) -> np.ndarray:
        """Calculate the torque vector based on the control vector.
        Computed through sum of the moments due to RCS thrusters, computed using the cross product of the engine positions and thrust vectors."""
        control_vector = self.controller.get_control_action()
        total_torque = np.zeros(3, 1)
        for i, (position, direction, max_thrust) in enumerate(
            zip(
                self._engine_positions,  # pylint: disable=no-member
                self._engine_directions,  # pylint: disable=no-member
                self._engine_max_thrusts,  # pylint: disable=no-member
            )
        ):
            thrust_vector = (
                (direction / norm(direction)) * control_vector[i] * max_thrust
            )
            total_torque += np.cross(position, thrust_vector)
        return total_torque

    @override
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft, and adds RCS thrusters."""
        self._acceleration_settings = super()._get_acceleration_settings()
        # self._acceleration_settings[self.name][self.name] = [
        #     acceleration.thrust_from_engine("ReactionControlSystem")
        # ]
        return self._acceleration_settings
