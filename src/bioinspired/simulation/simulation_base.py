"""Simulation Base module
This module provides the base class for all simulator classes.
"""

import time
import json
import logging
from typing import Callable
from abc import ABC, abstractmethod
from tudatpy.numerical_simulation import create_dynamics_simulator
from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import (
    dependent_variable,
    propagator,
    integrator,
)


class SimulationBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for simulator designs.
    Each Simulator design should inherit from this class and implement the
    required methods.
    """

    def __init__(
        self,
        dependent_variables_list=[],
    ):
        self._start_epoch: float = 0.0  # Start epoch of the simulation
        self._end_epoch: float = 100.0  # End epoch of the simulation

        self.global_frame_origin = "SSB"
        self.global_frame_orientation = "ECLIPJ2000"

        # Simulator owned
        self._body_model: SystemOfBodies = None
        self.get_body_model()
        self._integrator: integrator.IntegratorSettings = None
        self._get_integrator()

        # Externally owned
        self._propagator: propagator.PropagatorSettings = None
        self._propagator_list: list[Callable[[], propagator.PropagatorSettings]] = []
        self._termination_list: list[propagator.PropagationTerminationSettings] = []
        self._custom_termination_list: list[
            propagator.PropagationTerminationSettings
        ] = []

        self._dependent_variables = dependent_variables_list

    def add_dependent_variable(
        self, variable: dependent_variable.SingleDependentVariableSaveSettings
    ):
        """Add a dependent variable to the simulation."""
        if variable not in self._dependent_variables:
            self._dependent_variables.append(variable)

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
    def get_body_model(self) -> SystemOfBodies:
        """Return the body model object"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _dump_body_model(self) -> str:
        """Dump the body model to a string representation in a JSON format such that it can be saved to the database."""

        body_model = self.get_body_model()
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
        # if self._propagator is not None:
        #     return self._propagator

        propagators = []
        for propagator_func in self._propagator_list:
            propagators.extend(propagator_func())

        self._propagator = propagator.multitype(
            propagators,
            self._get_integrator(),
            self._start_epoch,
            self._get_termination_conditions(),
            output_variables=self._dependent_variables,
        )
        self._propagator.print_settings.print_initial_and_final_conditions = False
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
                == "propagator.PropagationDependentVariableTerminationSettings"
            ):
                # Create a dependent variable termination condition
                termination_settings = propagator.dependent_variable_termination(
                    condition, value
                )
                self._termination_list.append(termination_settings)
            elif condition_type == "propagator.PropagationTimeTerminationSettings":
                # Create a time termination condition
                termination_settings = propagator.time_termination(
                    value, terminate_exactly_on_final_condition=True
                )
                self._termination_list.append(termination_settings)
            elif condition_type == "propagator.PropagationCPUTimeTerminationSettings":
                # Create a CPU time termination condition
                termination_settings = propagator.cpu_time_termination(value)
                self._termination_list.append(termination_settings)
            elif condition_type == "propagator.PropagationCustomTerminationSettings":
                # Create a custom termination condition
                if callable(condition):
                    termination_settings = propagator.custom_termination(condition)
                    self._termination_list.append(termination_settings)
                else:
                    raise ValueError(
                        "Custom termination conditions must be callable functions."
                    )
            else:
                raise ValueError(
                    f"Unknown termination condition type: {condition_type}. "
                    "Supported types are: 'propagator.PropagationDependentVariableTerminationSettings', "
                    "'propagator.PropagationTimeTerminationSettings', "
                    "'propagator.PropagationCPUTimeTerminationSettings',"
                    "'propagator.PropagationCustomTerminationSettings'."
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
            "propagator.PropagationCustomTerminationSettings",
        ]:
            raise ValueError(
                "Invalid termination condition type. Supported types are: "
                "'propagator.PropagationDependentVariableTerminationSettings', "
                "'propagator.PropagationTimeTerminationSettings', "
                "'propagator.PropagationCPUTimeTerminationSettings',"
                "'propagator.PropagationCustomTerminationSettings'."
            )
        self._custom_termination_list.append(termination_condition)

    def dump_termination_conditions(self) -> str:
        """Dump the termination conditions to a string representation in a JSON format such that it can be saved to the database."""
        return json.dumps(self._custom_termination_list)

    def run(self, start_epoch: float, end_epoch: float):
        """Run the simulation"""
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._termination_list = []
        self.add_termination_condition(
            {
                "type": "propagator.PropagationTimeTerminationSettings",
                "condition": None,
                "value": self._end_epoch,
            }
        )

        # Telemetry: Display simulation start information with correct namespace
        logger = logging.getLogger(self.__class__.__module__)
        simulation_class = self.__class__.__name__
        logger.debug(f"{simulation_class}: Starting simulation")
        logger.debug(f"   Start epoch: {self._start_epoch:.2f}s")
        logger.debug(f"   End epoch: {self._end_epoch:.2f}s")
        logger.debug(f"   Duration: {self._end_epoch - self._start_epoch:.2f}s")
        start_time = time.time()
        result = create_dynamics_simulator(
            self.get_body_model(), self._get_propagators()
        )
        end_time = time.time()
        logger.debug(
            f"{simulation_class}: Simulation completed in {end_time - start_time:.2f}s"
        )
        # Return the result of the simulation
        return result
