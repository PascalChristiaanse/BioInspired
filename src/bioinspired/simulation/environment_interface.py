"""Environment Interface for Bio-Inspired Systems
This module defines the interface for the environment in a bio-inspired system.

"""


import numpy as np

class EnvironmentInterface:
    """
    Interface for the environment in a bio-inspired system.
    This interface defines the methods that any environment class should implement.
    """
    def get_acceleration(self, x: np.ndarray) -> np.ndarray:
        """
        Get the acceleration at a given position in the environment.
        
        :param x: An array representing the of a test particle in the environment.
        :type x: np.ndarray
        :return: An array representing the acceleration at the given position.
        :rtype: np.ndarray
        """
        raise NotImplementedError("This method should be overridden by subclasses.")