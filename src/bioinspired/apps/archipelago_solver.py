"""Archipelago Solver Application
This module uses the TrackedArchipelago class to run a distributed optimization
algorithm across multiple islands, tracking progress and individuals."""

import pygmo as pg

from bioinspired.problem import StopNeuronBasicProblem as Problem
from bioinspired.algorithm import (
    TrackedAlgorithm,
    TrackedArchipelago,
    SliceablePopulation,
)
from bioinspired.data import PyGMOBase
import logging
import os
import sys
import winsound


def setup_logging():
    """Set up logging configuration for the application."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    return logger


def main():
    """
    Main function to run the archipelago solver.
    Initializes the archipelago with specified parameters and runs the evolution process.
    """
    if sys.platform.startswith("win"):
        winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)    
    else:
        # For Unix-like systems, print the ASCII bell character
        print("\a")

    logger = setup_logging()
    logger.info("Starting Archipelago Solver")
    
    logger.info("Creating initial population")
    initial_population = SliceablePopulation(
        prob=Problem(stop_threshold=0.9),
        b=pg.mp_bfe(),
        size=8 * 16,  # Initial population size
        seed=42,  # Seed for reproducibility
    )
    logger.info(f"Initial population created with size: {len(initial_population)}")
    
    # Setup tables
    logger.info("Setting up tables for population tracking")
    PyGMOBase.create_tables()
    logger.info("Tables created successfully")
    
    
    # Create an instance of TrackedArchipelago with desired parameters
    logger.info("Initializing TrackedArchipelago")
    archipelago = TrackedArchipelago(
        num_islands=16,
        pop=initial_population,
        algo=pg.sga,
        algo_parameters={"cr": 0.95, "m": 0.01},  # JLeitner (2010)
        generations_per_migration_event=3,
        migration_events=5,  # Number of migration events
    )
    logger.info("TrackedArchipelago initialized successfully")
    
    # Start the evolution process
    logger.info("Starting evolution process")
    archipelago.evolve()
    logger.info("Evolution process completed")
    # Play a chime sound when the program finishes
    if sys.platform.startswith("win"):
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)    
    else:
        # For Unix-like systems, print the ASCII bell character
        print("\a")


if __name__ == "__main__":
    main()
