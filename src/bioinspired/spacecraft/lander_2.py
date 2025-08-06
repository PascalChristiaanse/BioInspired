"""Lander 2 spacecraft module.
This module defines the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
It is based on the paper
"""

import numpy as np

from overrides import override
from numpy.linalg import norm
from numba import jit

from tudatpy.numerical_simulation.propagation_setup import (
    acceleration,
    torque,
)

from bioinspired.simulation.simulation_base import SimulationBase

from .propelled_spacecraft_base import PropelledSpacecraftBase
from .rotating_spacecraft_base import RotatingSpacecraftBase
from .JSON_spacecraft_base import JSONSpacecraftBase

from bioinspired.controller import ControllerBase


@jit(nopython=True, cache=True)
def _compute_thrust_vector_jit(control_vector, engine_directions, engine_max_thrusts):
    """JIT-compiled helper function for thrust vector calculation."""
    total_thrust = np.zeros(3)
    for i in range(len(control_vector)):
        direction = engine_directions[i]
        max_thrust = engine_max_thrusts[i]
        direction_norm = np.sqrt(np.sum(direction * direction))
        if direction_norm > 0:
            normalized_direction = direction / direction_norm
            total_thrust += normalized_direction * control_vector[i] * max_thrust
    return total_thrust


@jit(nopython=True, cache=True)
def _compute_torque_vector_jit(
    control_vector, engine_positions, engine_directions, engine_max_thrusts
):
    """JIT-compiled helper function for torque vector calculation."""
    total_torque = np.zeros(3)
    for i in range(len(control_vector)):
        position = engine_positions[i]
        direction = engine_directions[i]
        max_thrust = engine_max_thrusts[i]

        direction_norm = np.sqrt(np.sum(direction * direction))
        if direction_norm > 0:
            normalized_direction = direction / direction_norm
            thrust_vector = normalized_direction * control_vector[i] * max_thrust
            # Cross product: position × thrust_vector
            torque_contribution = np.array(
                [
                    position[1] * thrust_vector[2] - position[2] * thrust_vector[1],
                    position[2] * thrust_vector[0] - position[0] * thrust_vector[2],
                    position[0] * thrust_vector[1] - position[1] * thrust_vector[0],
                ]
            )
            total_torque += torque_contribution
    return total_torque


class Lander2(RotatingSpacecraftBase, PropelledSpacecraftBase, JSONSpacecraftBase):
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
            initial_state=initial_state[:6],
        )
        self.controller: ControllerBase = controller
        self._torque_settings = {}

        # Cache engine data as NumPy arrays for JIT compilation
        self._engine_directions_array = None
        self._engine_max_thrusts_array = None
        self._engine_positions_array = None
        self._initialize_engine_arrays()

    def _initialize_engine_arrays(self):
        """Initialize cached NumPy arrays from engine data for JIT functions."""
        if self._engine_directions_array is None and hasattr(self, "_engines"):
            self._engine_directions_array = np.array(
                [engine["thrust_direction"] for engine in self._engines]  # pylint: disable=no-member
            )
            self._engine_max_thrusts_array = np.array(
                [engine["maximum_thrust"] for engine in self._engines]  # pylint: disable=no-member
            )
            self._engine_positions_array = np.array(
                [engine["location"] for engine in self._engines]  # pylint: disable=no-member
            )

    def get_orientation(self) -> np.ndarray:
        """Get the orientation of the Lander 2 spacecraft."""
        body_model = self._simulation.get_body_model().get(self.name)
        if hasattr(body_model, "body_fixed_to_inertial_frame"):
            return np.array(body_model.body_fixed_to_inertial_frame).flatten()

    @property
    @override
    def Isp(self) -> float:
        """Return the specific impulse (Isp) of the Lander 2 spacecraft."""
        return self._engines[0][
            "specific_impulse"
        ]  # Assuming all engines have the same Isp pylint: disable=no-member

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
            self.name: {self.name: [torque.custom_torque(self.get_torque_vector)]}
        }
        return self._torque_settings

    def get_thrust_vector(self, current_time: float) -> np.ndarray:
        """Calculate the thrust based on the control vector.
        Computed through sum of the forces due to RCS thrusters."""
        if current_time != current_time:
            return np.zeros((3, 1))
        control_vector = self.controller.get_control_action(current_time)

        # Use JIT-compiled function for performance
        if self._engine_directions_array is not None:
            return _compute_thrust_vector_jit(
                control_vector,
                self._engine_directions_array,
                self._engine_max_thrusts_array,
            ).reshape(3, 1)

        # Fallback to original implementation
        total_thrust = np.zeros((3, 1))
        for i, direction, max_thrust in enumerate(
            zip(self._engine_directions, self._engine_max_thrusts)  # pylint: disable=no-member
        ):
            total_thrust += (
                (direction / norm(direction)) * control_vector[i] * max_thrust
            )
        return total_thrust

    def get_torque_vector(self, current_time: float) -> np.ndarray:
        """Calculate the torque vector based on the control vector.
        Computed through sum of the moments due to RCS thrusters, computed using the cross product of the engine positions and thrust vectors."""
        control_vector = self.controller.get_control_action(current_time)

        # Use JIT-compiled function for performance
        if self._engine_positions_array is not None:
            total_torque = _compute_torque_vector_jit(
                control_vector,
                self._engine_positions_array,
                self._engine_directions_array,
                self._engine_max_thrusts_array,
            )
            return total_torque.reshape(3, 1)

        # Fallback to original implementation
        total_torque = np.zeros(3)
        for i, engine in enumerate(self._engines):  # pylint: disable=no-member)
            position = engine["location"]
            direction = engine["thrust_direction"]
            max_thrust = engine["maximum_thrust"]
            thrust_vector = (
                (direction / norm(direction)) * control_vector[i] * max_thrust
            )
            total_torque += np.cross(position, thrust_vector)
        return total_torque.reshape(3, 1)

    @override
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft, and adds RCS thrusters."""
        # print("Compiling acceleration settings for Lander 2 spacecraft...")
        self._acceleration_settings = super()._get_acceleration_settings()
        return self._acceleration_settings
