"""Spacecraft base class.
This module contains the base class for spacecraft designs in the bioinspired package.
It provides an interface for spacecraft designs and configurations such that each
spacecraft design can be used in the same way.
"""

import numpy as np
from abc import ABC, abstractmethod

from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import (
    acceleration,
    create_acceleration_models,
    propagator,
)

from bioinspired.simulation.simulation_base import SimulatorBase


class SpacecraftBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for spacecraft designs and configurations.
    Each spacecraft design should inherit from this class and implement the
    required methods.
    """

    @abstractmethod
    def __init__(self, name: str, simulation: SimulatorBase, initial_state: np.ndarray):
        """Initialize the spacecraft with a name."""
        self.name = name
        self._acceleration_model = None

        self._simulation = simulation
        self._acceleration_settings = None
        self._insert_into_body_model()
        self._simulation._propagator_list.append(
            self._get_propagator
        )  # This is a list of functions
        self._simulation._termination_list.append(
            self._get_termination()
        )  # This is a list of settings. Notice the difference in brackets
        if initial_state is not None and (
            initial_state.shape == (6, 1)
            or initial_state.shape == (1, 6)
            or initial_state.shape == (6,)
        ):
            self._initial_state = initial_state
        else:
            raise ValueError(
                "Initial state must be a 6-element array representing position and velocity."
            )

    def _get_acceleration_model(self):
        """Convert the acceleration settings into an acceleration model."""
        # Create the acceleration model for the spacecraft.
        self._acceleration_model = create_acceleration_models(
            self._simulation._get_body_model(),
            self._get_acceleration_settings(),
            [self.name],
            self._simulation._get_central_body(),
        )
        return self._acceleration_model

    @abstractmethod
    def _get_acceleration_settings(self) -> acceleration.AccelerationSettings:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft."""

        acceleration_dict = {}
        for body in self._simulation._get_body_model().list_of_bodies():
            if body == self.name:
                continue
            acceleration_dict[body.name] = [acceleration.point_mass_gravity()]

        # Create global accelerations dictionary.
        self._acceleration_settings = {self.name: acceleration_dict}
        return self._acceleration_settings

    def get_name(self) -> str:
        """Return the name of the spacecraft."""
        return self.name

    def _insert_into_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        if self._simulation._get_body_model().does_body_exist(self.name) is False:
            self._simulation._get_body_model().create_empty_body(self.name)
        else:
            raise ValueError(
                f"Body with name {self.name} already exists in the body model."
            )
        return self._simulation._body_model

    @abstractmethod
    def _get_propagator(self) -> propagator.PropagatorSettings:
        """Return the propagator settings object."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_termination(self) -> propagator.PropagationTerminationSettings:
        """Return the termination conditions for the spacecraft."""
        raise NotImplementedError("This method should be implemented by subclasses.")
