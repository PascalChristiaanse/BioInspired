"""Simulation Base module
This module provides the base class for all simulator classes.
"""

from abc import ABC, abstractmethod


class SimulatorBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for simulator designs.
    Each Simulator design should inherit from this class and implement the
    required methods.
    """

    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def _get_integrator(self):
        """Return the integrator settings object"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_body_model(self):
        """Return the body model object"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    @abstractmethod
    def _get_propagator(self):
        """Return the propagator settings object"""
        raise NotImplementedError("This method should be implemented by subclasses.")
