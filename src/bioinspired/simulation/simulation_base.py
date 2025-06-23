"""Simulation Base module
This module provides the base class for all simulator classes.
"""

import json
from typing import Callable
from abc import ABC, abstractmethod
from tudatpy.numerical_simulation import create_dynamics_simulator
from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import (
    propagator,
    integrator,
)


class SimulatorBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for simulator designs.
    Each Simulator design should inherit from this class and implement the
    required methods.
    """

    def __init__(self):
        super().__init__()
        self._start_epoch: float = 0.0  # Start epoch of the simulation
        self._end_epoch: float = 100.0  # End epoch of the simulation

        self.global_frame_origin = "SSB"
        self.global_frame_orientation = "ECLIPJ2000"

        # Simulator owned
        self._body_model: SystemOfBodies = None
        self._get_body_model()
        self._integrator: integrator.IntegratorSettings = None
        self._get_integrator()

        # Externally owned
        self._propagator: propagator.PropagatorSettings = None
        self._propagator_list: list[Callable[[], propagator.PropagatorSettings]] = []
        self._termination_list: list[propagator.PropagationTerminationSettings] = []
        self._custom_termination_list: list[
            propagator.PropagationTerminationSettings
        ] = []

    @abstractmethod
    def _get_central_body(self) -> list[str]:
        """Returns the simulator's central body.
        Must be implemented to guarantee a central body is set"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_integrator(self) -> integrator.IntegratorSettings:
        """Return the integrator settings object"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_body_model(self) -> SystemOfBodies:
        """Return the body model object"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _dump_body_model(self) -> str:
        """Dump the body model to a string representation in a JSON format such that it can be saved to the database."""

        body_model = self._get_body_model()
        list_of_bodies = body_model.list_of_bodies()

        # Create a minimal representation for database storage
        body_data = {
            "bodies": list_of_bodies,
            "central_bodies": self._get_central_body(),
        }

        return json.dumps(body_data)

    @abstractmethod
    def _dump_integrator_settings(self) -> str:
        """Dump the integrator settings to a string representation in a JSON format such that it can be saved to the database."""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _get_propagators(self) -> propagator.PropagatorSettings:
        """Compile the propagators for the simulation."""

        propagators = []
        for propagator_func in self._propagator_list:
            propagators.append(propagator_func())

        self._propagator = propagator.multitype(
            propagators,
            self._integrator,
            self._start_epoch,
            self._get_termination_conditions(),
            # output_variables=dependent_variables_to_save,
        )
        return self._propagator

    def _get_termination_conditions(self):
        """Return the termination condition for the simulation"""
        for termination in self._custom_termination_list:
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
                self._termination_list.append(termination_settings)
            elif condition_type == propagator.PropagationTimeTerminationSettings:
                # Create a time termination condition
                termination_settings = propagator.PropagationTimeTerminationSettings(
                    value
                )
                self._termination_list.append(termination_settings)
            elif condition_type == propagator.PropagationCPUTimeTerminationSettings:
                # Create a CPU time termination condition
                termination_settings = propagator.PropagationCPUTimeTerminationSettings(
                    value
                )
                self._termination_list.append(termination_settings)
            else:
                raise ValueError(
                    f"Unknown termination condition type: {condition_type}. "
                    "Supported types are: 'propagator.PropagationDependentVariableTerminationSettings', "
                    "'propagator.PropagationTimeTerminationSettings', "
                    "'propagator.PropagationCPUTimeTerminationSettings'."
                )
        termination_settings = propagator.hybrid_termination(
            self._termination_list, fulfill_single_condition=True
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
        self._custom_termination_list.append(termination_condition)

    def dump_termination_conditions(self) -> str:
        """Dump the termination conditions to a string representation in a JSON format such that it can be saved to the database."""
        return json.dumps(self._custom_termination_list)

    def run(self, start_epoch: float, simulation_time: float):
        """Run the simulation"""
        self._start_epoch = start_epoch
        self._end_epoch = start_epoch + simulation_time
        # Create simulation object and propagate dynamics.
        return create_dynamics_simulator(
            self._get_body_model(), self._get_propagators()
        )
