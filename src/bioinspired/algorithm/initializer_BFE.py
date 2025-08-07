"""Initializer BFE (Batch function evaluator)
This module provides the Batch Function Evaluator (BFE) for initializing
an archipelago in PyGMO. It is designed to work with the BasicProblem
and uses multiprocessing for efficient evaluation of an initial population."""

import numpy as np
import multiprocessing as mp
import pygmo as pg

import logging


# class InitializerBFE(pg.bfe):
class InitializerBFE:
    """Batch Function Evaluator for initializing an archipelago."""

    def __init__(self, dimensions, core_count=4):
        # super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"InitializerBFE initialized for {dimensions} dimensions")
        self.dimensions = dimensions
        self.core_count = max(core_count, 1) # Always use at least 1 core

    def __call__(self, prob, dvs: np.ndarray):
        dvs = np.reshape(dvs, (-1, self.dimensions))
        self.logger.info(
            f"Starting batch function evaluation for {dvs.shape[0]} decision vectors"
        )
        
        core_count = min(
            self.core_count, 2 * mp.cpu_count()
        )  # Use up to 2x oversubscription
        taskcount = min(core_count, len(dvs))  # Use up to 16 processes
        if core_count > mp.cpu_count():
            self.logger.debug(
                f"Employing {core_count} workers. Using oversubscription..."
            )
        else:
            self.logger.debug(f"Employing {taskcount} total workers")

        # Create a pool of workers
        with mp.Pool(processes=taskcount) as pool:
            # Map the dvs to the BasicProblem's fitness function
            results = pool.map(prob.fitness, dvs)

        # Convert results to a numpy array
        fitness = np.array(results, dtype=np.float64)
        self.logger.info(
            f"Batch function evaluation completed: {fitness.shape[0]} fitness vectors computed"
        )
        self.logger.debug(f"Fitness vectors: {fitness}")
        return fitness.flatten()

    def get_name(self):
        """Return the name of the initializer."""
        return "InitializerBFE"
