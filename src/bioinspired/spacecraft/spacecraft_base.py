"""Spacecraft base class.
This module contains the base class for spacecraft designs in the bioinspired package.
It provides an interface for spacecraft designs and configurations such that each
spacecraft design can be used in the same way.
"""

from abc import ABC, abstractmethod

from tudatpy.numerical_simulation.environment import SystemOfBodies

class SpacecraftBase(ABC):
    """Base class for spacecraft designs.

    This class provides an interface for spacecraft designs and configurations.
    Each spacecraft design should inherit from this class and implement the
    required methods.
    """

    @abstractmethod
    def __init__(self, name: str, body_model: SystemOfBodies):
        """Initialize the spacecraft with a name."""
        self.name = name
        self.insert_into_body_model(body_model)

    def get_name(self) -> str:
        """Return the name of the spacecraft."""
        return self.name

    def insert_into_body_model(self, body_model):
        """Self-inserts the spacecraft into a provided body model"""
        if body_model.get[self.name] is not None:
            raise ValueError(f"Spacecraft {self.name} already exists in the body model.")
        print("Inserting massless spacecraft into body model:", self.name)
        body_model.create_empty_body(self.name)
        
        
