"""Spacecraft base class.
This module contains the base class for spacecraft designs in the bioinspired package.
It provides an interface for spacecraft designs and configurations such that each
spacecraft design can be used in the same way.
"""

import json
import numpy as np
from abc import ABC, abstractmethod

from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import (
    acceleration,
    create_acceleration_models,
    propagator,
)

from bioinspired.simulation.simulation_base import SimulationBase


class SpacecraftBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for spacecraft designs and configurations.
    Each spacecraft design should inherit from this class and implement the
    required methods.
    """

    @abstractmethod
    def __init__(
        self, name: str, simulation: SimulationBase, initial_state: np.ndarray, **kwargs
    ):
        """Initialize the spacecraft with a name."""
        self.name = name
        self._acceleration_settings = None
        self._acceleration_model = None
        self._termination_settings: list[dict] = []

        self._simulation = simulation
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
            self._translational_state = initial_state
        else:
            raise ValueError(
                "Initial state must be a 6-element array representing position and velocity."
            )

    def _get_acceleration_model(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Convert the acceleration settings into an acceleration model."""
        # Create the acceleration model for the spacecraft.
        self._acceleration_model = create_acceleration_models(
            self._simulation.get_body_model(),
            self._get_acceleration_settings(),
            [self.name],
            self._simulation._get_central_body(),
        )
        return self._acceleration_model

    def dump_acceleration_settings(self) -> str:
        """Dump the acceleration settings to a string representation in a JSON format."""
        return json.dumps(self._acceleration_settings)

    @abstractmethod
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft."""

        acceleration_dict = {}
        for body in self._simulation.get_body_model().list_of_bodies():
            if body == self.name:
                continue
            acceleration_dict[body] = [acceleration.point_mass_gravity()]

        # Create global accelerations dictionary.
        self._acceleration_settings = {self.name: acceleration_dict}
        return self._acceleration_settings

    def get_name(self) -> str:
        """Return the name of the spacecraft."""
        return self.name

    def _insert_into_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        if self._simulation.get_body_model().does_body_exist(self.name) is False:
            self._simulation.get_body_model().create_empty_body(self.name)
        else:
            raise ValueError(
                f"Body with name {self.name} already exists in the body model."
            )
        return self._simulation._body_model

    @abstractmethod
    def _get_propagator(self) -> list[propagator.PropagatorSettings]:
        """Return the propagator settings object."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_termination(self):
        """Return the termination condition for the spacecraft"""
        termination_conditions = []
        for termination in self._termination_settings:
            if not isinstance(termination, dict):
                raise ValueError(
                    "Custom termination conditions must be dictionaries with the format: "
                    "{'type': 'propagator.PropagationTerminationSettings', 'condition': <condition>, 'value': <value>}"
                )
            # Unpack the termination condition
            condition_type = termination.get("type")
            condition = termination.get("condition")
            value = termination.get("value")
            if (
                condition_type
                == propagator.PropagationDependentVariableTerminationSettings
            ):
                # Create a dependent variable termination condition
                termination_settings = (
                    propagator.PropagationDependentVariableTerminationSettings(
                        condition, value
                    )
                )
                termination_conditions.append(termination_settings)
            elif condition_type == propagator.PropagationTimeTerminationSettings:
                # Create a time termination condition
                termination_settings = propagator.PropagationTimeTerminationSettings(
                    value
                )
                termination_conditions.append(termination_settings)
            elif condition_type == propagator.PropagationCPUTimeTerminationSettings:
                # Create a CPU time termination condition
                termination_settings = propagator.PropagationCPUTimeTerminationSettings(
                    value
                )
                termination_conditions.append(termination_settings)
            else:
                raise ValueError(
                    f"Unknown termination condition type: {condition_type}. "
                    "Supported types are: 'propagator.PropagationDependentVariableTerminationSettings', "
                    "'propagator.PropagationTimeTerminationSettings', "
                    "'propagator.PropagationCPUTimeTerminationSettings'."
                )
        termination_settings = propagator.hybrid_termination(
            termination_conditions, fulfill_single_condition=True
        )
        return termination_settings

    def add_termination_condition(self, termination_condition: dict):
        """Add a termination condition to the simulation. Must be a dictionary with the format:
        {
            "type": "propagator.PropagationTerminationSettings",
            "condition": <condition>,
            "value": <value>
        }

        where <condition> is a string representing the condition type and <value> is the value for that condition.
        This method allows for custom termination conditions to be added to the simulation, while still being able to add them
        to the database as a JSON string.
        """
        if not isinstance(termination_condition, dict):
            raise ValueError(
                "Termination condition must be a dictionary with the format: "
                "{'type': 'propagator.PropagationTerminationSettings', 'condition': <condition>, 'value': <value>}"
            )
        # Validate the termination condition type
        if termination_condition["type"] not in [
            "propagator.PropagationDependentVariableTerminationSettings",
            "propagator.PropagationTimeTerminationSettings",
            "propagator.PropagationCPUTimeTerminationSettings",
        ]:
            raise ValueError(
                "Invalid termination condition type. Supported types are: "
                "'propagator.PropagationDependentVariableTerminationSettings', "
                "'propagator.PropagationTimeTerminationSettings', "
                "'propagator.PropagationCPUTimeTerminationSettings'."
            )
        self._termination_settings.append(termination_condition)

    def dump_termination_settings(self) -> str:
        """Dump the termination settings to a string representation in a JSON format."""
        return json.dumps(self._termination_settings)
