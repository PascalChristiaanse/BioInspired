"""Simulation Base module
This module provides the base class for all simulator classes.
"""

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
        self._propagator_list: list[callable[[], [propagator.PropagatorSettings]]] = []
        self._termination_list: list[propagator.TerminationCondition] = []

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
        termination_settings = propagator.hybrid_termination(
            self._termination_list, fulfill_single_condition=True
        )
        return termination_settings

    def run(self, start_epoch: float, simulation_time: float):
        """Run the simulation"""
        self._start_epoch = start_epoch
        self._end_epoch = start_epoch + simulation_time
        # Create simulation object and propagate dynamics.
        return create_dynamics_simulator(
            self._get_body_model(), self._get_propagators()
        )
