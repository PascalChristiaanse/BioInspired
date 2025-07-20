"""Rotating Spacecraft Base Module

This module provides a base class for spacecraft designs that include rotational dynamics.
It extends the SpacecraftBase class to include rotational properties and methods.
"""

import numpy as np
from abc import abstractmethod

from .spacecraft_base import SpacecraftBase

from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.numerical_simulation.propagation_setup import (
    torque,
    propagator,
)


class RotatingSpacecraftBase(SpacecraftBase):
    """Base class for spacecraft designs with rotational dynamics.
    This class extends the SpacecraftBase class to include rotational properties and methods.
    Each rotating spacecraft design should inherit from this class and implement the required methods.
    """

    def __init__(self, initial_rotational_state, **kwargs):
        """Initialize the rotating spacecraft with a name and initial state."""
        if initial_rotational_state.shape != (7,):
            raise ValueError(
                f"initial_rotational_state must be a 7-element vector. (got {initial_rotational_state.shape}, requires q0, q1, q2, q3, omega_x, omega_y, omega_z)"
            )
        super().__init__(**kwargs)
        self._rotational_state = initial_rotational_state
        self._set_rigid_body_properties()

    @property
    @abstractmethod
    def mass(self) -> float:
        """Abstract property for spacecraft mass."""
        raise NotImplementedError("Subclasses must implement the mass property.")

    @property
    @abstractmethod
    def inertia_tensor(self) -> np.ndarray[np.ndarray]:
        """Abstract property for spacecraft inertia tensor."""
        raise NotImplementedError(
            "Subclasses must implement the inertia_tensor property."
        )

    @property
    @abstractmethod
    def center_of_mass(self) -> np.ndarray:
        """Abstract property for spacecraft center of mass."""
        raise NotImplementedError(
            "Subclasses must implement the center_of_mass property."
        )

    def _set_rigid_body_properties(self):
        """Set the rigid body properties of the spacecraft using abstract properties."""
        settings = environment_setup.rigid_body.constant_rigid_body_properties(
            mass=self.mass,
            inertia_tensor=self.inertia_tensor,
            center_of_mass=self.center_of_mass,
        )
        environment_setup.add_rigid_body_properties(
            self._simulation.get_body_model(),
            self.name,
            settings,
        )

    @abstractmethod
    def _get_torque_settings(
        self,
    ) -> dict[str, dict[str, list[torque.TorqueSettings]]]:
        """Compiles the torque model for the spacecraft."""
        raise NotImplementedError(
            "Subclasses must implement the _get_torque_settings method."
        )

    def _get_torque_models(self):
        """Returns the torque models for the spacecraft."""
        return propagation_setup.create_torque_models(
            self._simulation.get_body_model(),
            self._get_torque_settings(),
            [self.name],
        )

    def _get_propagator(self) -> list[propagator.PropagatorSettings]:
        """Return the propagator settings for the spacecraft."""
        # Create a propagator settings object for the spacecraft
        rotational_propagator_settings = propagator.rotational(
            self._get_torque_models(),
            [self.name],
            self._rotational_state,
            self._simulation._start_epoch,
            self._simulation._get_integrator(),
            self._get_termination(),
            # output_variables=dependent_variables_to_save,
        )
        translational_propagator_settings = propagator.translational(
            self._simulation._get_central_body(),
            self._get_acceleration_model(),
            [self.name],
            self._translational_state,
            self._simulation._start_epoch,
            self._simulation._get_integrator(),
            self._get_termination(),
            # output_variables=dependent_variables_to_save
        )
        return [translational_propagator_settings, rotational_propagator_settings]
