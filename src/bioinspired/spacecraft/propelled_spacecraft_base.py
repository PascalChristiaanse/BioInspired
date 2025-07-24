"""Propelled Spacecraft Base Module
This module contains the base class for propelled spacecraft designs that implements engine models.
"""

import numpy as np

from abc import ABC, abstractmethod
from numba import jit
from bioinspired.spacecraft.spacecraft_base import SpacecraftBase

from tudatpy.numerical_simulation.environment_setup import (
    rigid_body,
    add_rigid_body_properties,
    add_variable_direction_engine_model,
)
from tudatpy.numerical_simulation.propagation_setup import thrust
from tudatpy.numerical_simulation.propagation_setup import acceleration


@jit(nopython=True, cache=True)
def _compute_thrust_direction_jit(thrust_vector):
    """JIT-compiled helper function for thrust direction calculation."""
    thrust_norm = np.sqrt(np.sum(thrust_vector * thrust_vector))
    if thrust_norm == 0:
        return np.zeros(3)
    return thrust_vector / thrust_norm


@jit(nopython=True, cache=True)
def _compute_thrust_magnitude_jit(thrust_vector):
    """JIT-compiled helper function for thrust magnitude calculation."""
    return np.sqrt(np.sum(thrust_vector * thrust_vector))


class PropelledSpacecraftBase(SpacecraftBase):
    """Base class for propelled spacecraft designs.
    This class extends the SpacecraftBase class to include propulsion capabilities.
    It provides an interface for spacecraft designs that implement engine models.
    Each propelled spacecraft design should inherit from this class and implement the
    required methods.
    """

    def __init__(self, **kwargs):
        """Initialize the propelled spacecraft with a name and initial state."""
        super().__init__(**kwargs)
        self._set_rigid_body_properties()
        self._setup_engine_model()

    @property
    @abstractmethod
    def Isp(self) -> float:
        """Abstract property for specific impulse (Isp) of the spacecraft."""
        raise NotImplementedError("Subclasses must implement the Isp property.")

    @property
    @abstractmethod
    def mass(self) -> float:
        """Abstract property for spacecraft mass."""
        raise NotImplementedError("Subclasses must implement the mass property.")

    @abstractmethod
    def get_thrust_vector(self, current_time: float) -> np.ndarray:
        """Calculate the thrust vector based on the engine model.
        This method should be implemented by subclasses to return the specific thrust vector.
        """
        raise NotImplementedError(
            "Subclasses must implement the get_thrust_vector method."
        )

    def _set_rigid_body_properties(self):
        """Set the rigid body properties of the spacecraft using abstract properties."""
        settings = rigid_body.constant_rigid_body_properties(mass=self.mass)
        add_rigid_body_properties(
            self._simulation.get_body_model(),
            self.name,
            settings,
        )

    def get_thrust_direction(self, current_time: float) -> np.ndarray:
        """Calculate the thrust direction based on the control vector."""
        thrust_vector = self.get_thrust_vector(current_time)
        thrust_vector_flat = thrust_vector.flatten()

        # Use JIT-compiled function for performance
        direction = _compute_thrust_direction_jit(thrust_vector_flat)
        return direction.reshape(3, 1)

    def get_thrust_magnitude(self, current_time: float) -> float:
        """Calculate the total thrust magnitude based on the control vector."""
        if current_time == current_time:
            thrust_vector = self.get_thrust_vector(current_time)
            thrust_vector_flat = thrust_vector.flatten()

            # Use JIT-compiled function for performance
            return _compute_thrust_magnitude_jit(thrust_vector_flat)
        # If no computation is to be done, return zeros
        else:
            return 0.0

    def _setup_engine_model(self):
        """Return the engine model for the spacecraft.
        This method should be implemented by subclasses to return the specific engine model.
        """
        thrust_magnitude_settings = thrust.custom_thrust_magnitude_fixed_isp(
            self.get_thrust_magnitude, self.Isp
        )
        add_variable_direction_engine_model(
            self.name,
            "ReactionControlSystem",
            thrust_magnitude_settings,
            self._simulation.get_body_model(),
            self.get_thrust_direction,
        )

    @abstractmethod
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for the spacecraft.
        This method should be implemented by subclasses to return the specific acceleration settings.
        """
        print("Compiling acceleration settings for propelled spacecraft...")
        self._acceleration_settings = super()._get_acceleration_settings()
        # Add thrust from the engine model
        self._acceleration_settings[self.name][self.name] = [
            acceleration.thrust_from_engine("ReactionControlSystem")
        ]
        return self._acceleration_settings
