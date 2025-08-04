"""Cost functions module.
This module contains various cost functions for use in the automatic rendezvous and docking (AR&D) problem posed in Interstellar (2014).
References:
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from overrides import override

from tudatpy.numerical_simulation.propagation import SingleArcSimulationResults


class CostFunctionBase(ABC):
    """Base class for cost functions.
    This class defines the interface for cost functions used in the AR&D problem.
    """

    @abstractmethod
    def cost(self, dynamics_simulator: SingleArcSimulationResults) -> float:
        """Calculate the cost for the agent's trajectory relative to the target trajectory.
        This method should be overridden by subclasses to implement specific cost calculations.
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
        self.t_max = 1000  # maximum time allowed for docking procedure, same as in J. Leitner et al. (2010)
        self.docking_tolerance_position = 0.1  # meters
        self.docking_tolerance_velocity = 0.1
        self.docking_tolerance_orientation = np.pi / 8

    @override
    def cost(self, dynamics_simulator: SingleArcSimulationResults) -> float:
        """Cost function based on rendevous approach and docking requirements."""

        final_time = max(target_trajectory.keys())

        # Calculate constraints
        # Orientation error
        target_orientation = target_trajectory[final_time][
            6:10
        ]  # Quaternion orientation
        agent_orientation = agent_trajectory[final_time][6:10]
        # Compute angular difference
        orientation_error = np.arccos(
            np.clip(np.dot(target_orientation, agent_orientation), -1.0, 1.0)
        )

        # Distance error
        target_position = target_trajectory[final_time][:3]
        agent_position = agent_trajectory[final_time][:3]
        position_error = np.linalg.norm(target_position - agent_position)

        # Velocity error
        target_velocity = target_trajectory[final_time][3:6]
        agent_velocity = agent_trajectory[final_time][3:6]
        velocity_error = np.linalg.norm(target_velocity - agent_velocity)

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
    def cost(self, dynamics_simulator: SingleArcSimulationResults) -> float:
        """Cost function without stop neuron logic."""
        self.t_max = self.determine_stop_time(target_trajectory, agent_trajectory)
        return super().cost(target_trajectory, agent_trajectory)

    def determine_stop_time(
        self, target_trajectory: dict[np.ndarray], agent_trajectory: dict[np.ndarray]
    ) -> float | None:
        """Determine the earliest time at which the docking constraints are met.

        Args:
        target_trajectory: Dictionary with time keys and state arrays as values for target
        agent_trajectory: Dictionary with time keys and state arrays as values for agent

        Returns:
        The earliest time when all docking constraints are satisfied, or None if never satisfied
        """
        # Get all time keys that exist in both trajectories, sorted in ascending order
        common_times = sorted(
            set(target_trajectory.keys()) & set(agent_trajectory.keys())
        )

        for time in common_times:
            # Calculate constraints at this time step
            # Orientation error
            target_orientation = target_trajectory[time][6:10]  # Quaternion orientation
            agent_orientation = agent_trajectory[time][6:10]
            # Compute angular difference
            orientation_error = np.arccos(
                np.clip(np.dot(target_orientation, agent_orientation), -1.0, 1.0)
            )

            # Distance error
            target_position = target_trajectory[time][:3]
            agent_position = agent_trajectory[time][:3]
            position_error = np.linalg.norm(target_position - agent_position)

            # Velocity error
            target_velocity = target_trajectory[time][3:6]
            agent_velocity = agent_trajectory[time][3:6]
            velocity_error = np.linalg.norm(target_velocity - agent_velocity)

            # Check if all constraints are met at this time
            if (
                orientation_error < self.docking_tolerance_orientation
                and position_error < self.docking_tolerance_position
                and velocity_error < self.docking_tolerance_velocity
            ):
                return time

        # If no time satisfies all constraints, return final time
        return max(target_trajectory.keys())
