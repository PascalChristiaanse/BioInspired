"""Problem base class for PyGMO problems.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bioinspired.problem import CostFunctionBase

from abc import ABC, abstractmethod


class ProblemBase(ABC):
    """Base class for PyGMO problems.
    This class provides a common interface for problems in the bioinspired domain.
    It can be extended to implement specific problems like the automatic rendezvous and docking (AR&D) problem.
    """
    def __init__(self, cost_function: CostFunctionBase):
        super().__init__()
        self.cost_function = cost_function
        
    @abstractmethod
    def fitness(self, x):
        """Evaluate the fitness of a solution.
        This method should be overridden by subclasses to implement specific fitness calculations.
        
        Args:
            x (list): A list representing the solution in design space to evaluate.
        
        Returns:
            list: A list containing the fitness values for the solution.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def get_bounds(self):
        """Get the bounds of the problem.
        This method should be overridden by subclasses to provide the bounds for the decision variables.
        
        Returns:
            tuple: A tuple containing two lists, lower and upper bounds for the decision variables.
        """
        raise NotImplementedError("Subclasses must implement this method.")
        
    def get_name(self):
        """Get the name of the problem.
        This method can be overridden by subclasses to provide a specific name for the problem.
        
        Returns:
            str: The name of the problem.
        """
        return self.__class__.__name__