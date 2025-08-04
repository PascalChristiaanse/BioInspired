"""Problem base class for PyGMO problems.
"""

from abc import ABC, abstractmethod


class ProblemBase(ABC):
    """Base class for PyGMO problems.
    This class provides a common interface for problems in the bioinspired domain.
    It can be extended to implement specific problems like the automatic rendezvous and docking (AR&D) problem.
    """
    def __init__(self):
        super().__init__()
        
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
        
    