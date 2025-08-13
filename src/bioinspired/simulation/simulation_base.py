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
        self.global_frame_orientation = "ECLIPJ2000"  # Simulator owned
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
        # Simply add all custom termination conditions to the list
        for termination_setting in self._custom_termination_list:
            self._termination_list.append(termination_setting)

        termination_settings = propagator.hybrid_termination(
            self._termination_list, fulfill_single_condition=True
        )
        return termination_settings

    def add_termination_condition(self, termination_condition):
        """Add a termination condition to the simulation.

        Can accept either:
        1. A dictionary with the format:
           {
               "type": "propagator.PropagationTerminationSettings",
               "condition": <condition>,
               "value": <value>
           }
        2. A direct propagator.PropagationTerminationSettings object

        This method allows for both JSON-serializable conditions and direct termination settings.
        """
        if hasattr(termination_condition, "__class__") and "propagator" in str(
            type(termination_condition)
        ):
            # Direct termination setting object
            self._custom_termination_list.append(termination_condition)
        elif isinstance(termination_condition, dict):
            # Dictionary format - convert to termination setting
            condition_type = termination_condition.get("type")
            condition = termination_condition.get("condition")
            value = termination_condition.get("value")

            if (
                condition_type
                == "propagator.PropagationDependentVariableTerminationSettings"
            ):
                termination_settings = propagator.dependent_variable_termination(
                    condition, value
                )
            elif condition_type == "propagator.PropagationTimeTerminationSettings":
                termination_settings = propagator.time_termination(
                    value, terminate_exactly_on_final_condition=True
                )
            elif condition_type == "propagator.PropagationCPUTimeTerminationSettings":
                termination_settings = propagator.cpu_time_termination(value)
            elif condition_type == "propagator.PropagationCustomTerminationSettings":
                # Assuming condition is a callable function for custom termination
                termination_settings = propagator.custom_termination(condition)
            else:
                raise ValueError(
                    f"Unknown termination condition type: {condition_type}. "
                    "Supported types are: 'propagator.PropagationDependentVariableTerminationSettings', "
                    "'propagator.PropagationTimeTerminationSettings', "
                    "'propagator.PropagationCPUTimeTerminationSettings',"
                    "'propagator.PropagationCustomTerminationSettings'."
                )
            self._custom_termination_list.append(termination_settings)
        else:
            raise ValueError(
                "Termination condition must be either a dictionary or a propagator.PropagationTerminationSettings object"
            )

    def dump_termination_conditions(self) -> str:
        """Dump the termination conditions to a string representation in a JSON format
        such that it can be saved to the database.

        Decomposes termination condition functions by their name and parameters for
        JSON serialization compatibility.
        """
        serializable_conditions = []

        for condition in self._custom_termination_list:
            condition_data = {}

            # Get the type/class name of the termination condition
            condition_type = type(condition).__name__
            condition_data["type"] = condition_type

            # Try to extract parameters based on common termination condition types
            if hasattr(condition, "termination_variable_"):
                # Dependent variable termination
                condition_data["condition"] = str(condition.termination_variable_)
                if hasattr(condition, "limit_value_"):
                    condition_data["value"] = condition.limit_value_
                if hasattr(condition, "use_as_lower_limit_"):
                    condition_data["use_as_lower_limit"] = condition.use_as_lower_limit_
                if hasattr(condition, "terminate_exactly_on_final_condition_"):
                    condition_data["terminate_exactly_on_final_condition"] = (
                        condition.terminate_exactly_on_final_condition_
                    )

            elif hasattr(condition, "final_time_"):
                # Time termination
                condition_data["value"] = condition.final_time_
                if hasattr(condition, "terminate_exactly_on_final_condition_"):
                    condition_data["terminate_exactly_on_final_condition"] = (
                        condition.terminate_exactly_on_final_condition_
                    )

            elif hasattr(condition, "cpu_time_limit_"):
                # CPU time termination
                condition_data["value"] = condition.cpu_time_limit_

            elif hasattr(condition, "custom_termination_condition_"):
                # Custom termination
                condition_data["function_name"] = getattr(
                    condition.custom_termination_condition_,
                    "__name__",
                    "unknown_function",
                )
                condition_data["description"] = "Custom termination function"

            else:
                # Generic fallback - try to extract any available attributes
                condition_data["description"] = (
                    f"Termination condition of type {condition_type}"
                )
                # Extract common attributes if they exist
                for attr in ["value", "condition", "limit", "threshold"]:
                    if hasattr(condition, attr):
                        condition_data[attr] = getattr(condition, attr)

            serializable_conditions.append(condition_data)

        return json.dumps(serializable_conditions, indent=2)

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
