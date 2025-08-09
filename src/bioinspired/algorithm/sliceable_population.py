"""Sliceable Population module

This module provides a custom population class that extends PyGMO's population
with slicing capabilities while maintaining all original functionality.
"""

import numpy as np
import pygmo as pg
from typing import Union


class SliceablePopulation(pg.population):
    """
    A PyGMO population with added slicing capabilities.
    
    This class inherits from pygmo.population and adds the ability to slice
    the population using standard Python slicing notation (e.g., pop[0:10]).
    All original PyGMO population methods are preserved.
    
    Examples:
        # Create a sliceable population
        pop = SliceablePopulation(problem, size=100)
        
        # Get first 10 individuals
        sub_pop = pop[0:10]
        
        # Get last 5 individuals  
        sub_pop = pop[-5:]
        
        # Get every other individual
        sub_pop = pop[::2]
        
        # Get a single individual
        single_pop = pop[5]
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the sliceable population with the same arguments as pygmo.population."""
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, key: Union[int, slice]) -> 'SliceablePopulation':
        """
        Enable slicing and indexing of the population.
        
        Args:
            key: Integer index or slice object
            
        Returns:
            SliceablePopulation: A new population containing the selected individuals
            
        Raises:
            IndexError: If index is out of range
            TypeError: If key is not int or slice
        """
        if isinstance(key, int):
            # Handle single index
            if key < 0:
                key += len(self)
            if not 0 <= key < len(self):
                raise IndexError(f"Population index {key} out of range for population of size {len(self)}")
            
            # Create a new population with just this individual
            new_pop = SliceablePopulation(self.problem)
            
            # Get the chromosome and fitness for this individual
            chromosomes = self.get_x()
            fitnesses = self.get_f()
            
            chromosome = chromosomes[key]
            fitness = fitnesses[key]
            
            # Add the individual to the new population
            new_pop.push_back(chromosome, fitness)
            
            return new_pop
            
        elif isinstance(key, slice):
            # Handle slice
            start, stop, step = key.indices(len(self))
            
            # Create a new population
            new_pop = SliceablePopulation(self.problem)
            
            # Get all chromosomes and fitnesses
            chromosomes = self.get_x()
            fitnesses = self.get_f()
            
            # Add the selected individuals to the new population
            for i in range(start, stop, step):
                chromosome = chromosomes[i]
                fitness = fitnesses[i]
                new_pop.push_back(chromosome, fitness)
            
            return new_pop
            
        else:
            raise TypeError(f"Population indices must be integers or slices, not {type(key).__name__}")
    
    def __len__(self) -> int:
        """Return the number of individuals in the population."""
        return self.get_x().shape[0]

    
    def __repr__(self) -> str:
        """String representation of the sliceable population."""
        return f"SliceablePopulation(size={len(self)}, problem={self.problem.get_name()})"
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return self.__repr__()
