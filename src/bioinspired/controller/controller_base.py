"""Controller base class
This module defines an interface for spacecraft controllers.
"""

import numpy as np

from abc import ABC, abstractmethod


class ControllerBase(ABC):
    """Base class for spacecraft controllers.
    This class provides an interface for controllers that can take a state vector and return a control action.
    """

    def __init__(
        self, simulator, lander_name: str, target_name: str = None
    ):
        """Initialize the controller."""
        self.simulator = simulator
        self.lander_name = lander_name
        self.target_name = target_name

    @abstractmethod
    def get_control_action(self, current_time):
        """Get the control action based on the control law implemented and the  current state of the simulation."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def extract_state_vector(self, current_time) -> np.ndarray:
        """
        Extracts a state vector from the simulation for neural net input.

        Returns:
            state_vector (np.ndarray): Flat vector of relevant state information
        """
        # Example: extract position and velocity of lander
        lander_body = self.simulator.get_body_model().get(self.lander_name)
        lander_state = (
            lander_body.state if hasattr(lander_body, "state") else np.zeros(6)
        )
        state_vector = [*lander_state]
    
        target_body = self.simulator.get_body_model().get(self.target_name)
        target_state = target_body.ephemeris.cartesian_state(current_time)
        state_vector.extend(target_state)

        # target distance
        distance = np.linalg.norm(
            np.array(lander_state[:3]) - np.array(target_state[:3])
        )
        state_vector.append(distance)

        # Lander orientation towards target
        relative_position = np.array(target_state[:3]) - np.array(lander_state[:3])
        state_vector.extend(relative_position)
        lander_velocity = np.array(lander_state[3:6])
        if (
            np.linalg.norm(lander_velocity) > 0
            and np.linalg.norm(relative_position) > 0
        ):
            cos_angle = np.dot(lander_velocity, relative_position) / (
                np.linalg.norm(lander_velocity) * np.linalg.norm(relative_position)
            )
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angular_separation = np.arccos(cos_angle)
        else:
            angular_separation = 0.0
        state_vector.append(angular_separation)

        # Difference in angular orientation using body_fixed_to_inertial_frame (rotation matrix or quaternion)
        lander_orientation = getattr(
            lander_body, "body_fixed_to_inertial_frame", np.zeros(9)
        )
        target_orientation = target_body.rotation_model.body_fixed_to_inertial_rotation(current_time)
        orientation_diff = lander_orientation - target_orientation
        state_vector.extend(orientation_diff.flatten())

        # Difference in angular rate using body_fixed_to_inertial_frame_derivative
        lander_angular_rate = getattr(
            lander_body, "body_fixed_to_inertial_frame_derivative", np.zeros(9)
        )
        target_angular_rate = target_body.rotation_model.time_derivative_body_fixed_to_inertial_rotation(current_time)
        angular_rate_diff = lander_angular_rate - target_angular_rate
        state_vector.extend(angular_rate_diff.flatten())
        return np.zeros(36) 
        return np.array(state_vector)
