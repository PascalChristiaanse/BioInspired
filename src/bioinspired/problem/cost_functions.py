"""Cost functions module.
This module contains various cost functions for use in the automatic rendezvous and docking (AR&D) problem posed in Interstellar (2014).
References:
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from overrides import override

from tudatpy.numerical_simulation.propagation import SingleArcSimulationResults
from scipy.spatial.transform import Rotation as R


class CostFunctionBase(ABC):
    """Base class for cost functions.
    This class defines the interface for cost functions used in the AR&D problem.
    """

    @abstractmethod
    def cost(self, parameters: dict[str, dict[float, np.array]]) -> float:
        """Calculate the cost for the agent's trajectory relative to the target trajectory.
        This method should be overridden by subclasses to implement specific cost calculations.
        The parameter dictionary should contain the name of each parameter as a key,
        and a dictionary with time keys and state vectors as values.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class JLeitner2010(CostFunctionBase):
    """Cost function based on Leitner, J., Ampatzis, C., & Izzo, D. (2010).
    Evolving ANNs for spacecraft rendezvous and docking. In Proceedings of the 10th International Symposium on Artificial Intelligence, Robotics and Automation in Space,
    i-SAIRAS 2010 (pp. 386-393). European Space Agency (ESA).

    This paper simulates the limited Hill-Clohessey-Wiltshire (HCW) problem using RK4.
    It also implements a simulation stop neuron, so the neural net is able to stop the simulation when it has decided it has docked.
    """

    def __init__(self):
        super().__init__()
        self.t_max = 50  # maximum time allowed for docking procedure, same as in J. Leitner et al. (2010)
        self.docking_tolerance_position = 0.1  # meters
        self.docking_tolerance_velocity = 0.1
        self.docking_tolerance_orientation = np.pi / 8

    @override
    def cost(self, parameters: dict[str, dict[float, np.array]]) -> float:
        """Cost function based on rendevous approach and docking requirements."""

        required_keys = [
            "relative_distance",
            "relative_speed",
            "orientation_matrix_A",
            "orientation_matrix_B",
        ]

        for key in required_keys:
            if key not in parameters:
                raise ValueError(f"Missing required key: {key}")

        final_time = max(parameters["relative_distance"].keys())

        # Get state error at final time
        position_error = parameters["relative_distance"][final_time]
        velocity_error = parameters["relative_speed"][final_time]

        # Get orientation error at final time
        final_orientation_A = parameters["orientation_matrix_A"][final_time]
        final_orientation_B = parameters["orientation_matrix_B"][final_time]

        # Convert rotation matrices to Euler angles (e.g., 'xyz' convention)
        euler_A = R.from_matrix(final_orientation_A).as_euler("xyz")
        euler_B = R.from_matrix(final_orientation_B).as_euler("xyz")

        # Compute orientation error as norm of angle difference
        orientation_error = np.linalg.norm(euler_A - euler_B)

        # Compute constraints
        if (
            orientation_error < self.docking_tolerance_orientation
            and position_error < self.docking_tolerance_position
            and velocity_error < self.docking_tolerance_velocity
        ):
            constraints_met = True
        else:
            constraints_met = False

        # Calculate cost
        if constraints_met:
            return 1 + (self.t_max - final_time) / self.t_max
        else:
            return 1 / (
                (1 + orientation_error) * (1 + position_error) * (1 + velocity_error)
            )


class JLeitner2010NoStopNeuron(JLeitner2010):
    """Cost function based on Leitner, J., Ampatzis, C., & Izzo, D. (2010).
    Evolving ANNs for spacecraft rendezvous and docking, without the stop neuron.
    This is a variant of the JLeitner2010 cost function that does not include the stop neuron logic.
    """

    @override
    def cost(self, parameters: dict[str, dict[float, np.array]]) -> float:
        """Cost function without stop neuron logic."""
        self.t_max = self.determine_stop_time(parameters)
        return super().cost(parameters)

    def determine_stop_time(
        self, parameters: dict[str, dict[float, np.array]]
    ) -> float | None:
        """Determine the earliest time at which the docking constraints are met.

        Args:
        target_trajectory: Dictionary with time keys and state arrays as values for target
        agent_trajectory: Dictionary with time keys and state arrays as values for agent

        Returns:
        The earliest time when all docking constraints are satisfied, or None if never satisfied
        """
        # Get all time keys that exist in both trajectories, sorted in ascending order

        for time in parameters["relative_distance"].keys():
            # Calculate constraints at this time step

            # Get state error at final time
            position_error = parameters["relative_distance"][time]
            velocity_error = parameters["relative_speed"][time]

            # Get orientation error at final time
            final_orientation_A = parameters["orientation_matrix_A"][time]
            final_orientation_B = parameters["orientation_matrix_B"][time]

            # Convert rotation matrices to Euler angles (e.g., 'xyz' convention)
            euler_A = R.from_matrix(final_orientation_A).as_euler("xyz")
            euler_B = R.from_matrix(final_orientation_B).as_euler("xyz")

            # Compute orientation error as norm of angle difference
            orientation_error = np.linalg.norm(euler_A - euler_B)

            # Check if all constraints are met at this time
            if (
                orientation_error < self.docking_tolerance_orientation
                and position_error < self.docking_tolerance_position
                and velocity_error < self.docking_tolerance_velocity
            ):
                return time

        # If no time satisfies all constraints, return final time
        return max(parameters["relative_distance"].keys())
