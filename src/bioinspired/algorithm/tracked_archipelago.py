"""
Simplified PyGMO archipelago with database tracking.
Minimal implementation focusing on basic database operations.
Includes minimal multiprocessing communication for cross-process algorithm tracking.
"""

import time
import logging
import numpy as np
import multiprocessing as mp
from typing import Optional, Dict
from sqlalchemy.orm import Session

import pygmo as pg
from bioinspired.data.database import get_session_context
from bioinspired.data.pygmo_models import Archipelago, Island, Individual, STATUS, PyGMOBase
from bioinspired.algorithm import SliceablePopulation, TrackedAlgorithm


class TrackedArchipelago:
    """
    Simplified wrapper around PyGMO archipelago with basic database tracking.
    Provides the same interface as pg.archipelago but with minimal database integration.
    """

    def __init__(
        self,
        num_islands: int,
        pop: SliceablePopulation,
        algo: pg.algorithm,
        algo_parameters: Optional[Dict] = None,
        generations_per_migration_event: int = 5,
        migration_events: int = 1,
        seed: int = 42,
        topology: pg.topology = None,
        track_champions_only: bool = True,
    ):
        """
        Initialize tracked archipelago.

        Args:
            topology: PyGMO topology (defaults to fully connected)
            track_champions_only: Whether to store only champions or all individuals
        """
        # Set a logger for this class
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Initializing TrackedArchipelago")

        self.num_islands = num_islands
        self.track_champions_only = track_champions_only
        self.algo_type = algo
        self.migration_events = migration_events

        self.archipelago_record: Archipelago = Archipelago(
            name="Tracked Archipelago",
            problem_class=pop.problem.get_name(),
            num_islands=num_islands,
            population_per_island=len(pop) / num_islands,
            generations_per_migration_event=generations_per_migration_event,
            migration_events=migration_events,
            total_generations=generations_per_migration_event * (migration_events + 1),
            seed=seed,
            description="Simplified tracked archipelago with database integration",
            status=STATUS.INITIALIZED.value,
        )
        self.archipelago_record.save_to_db()

        # Cast population to slicable population
        if type(pop) is not SliceablePopulation:
            raise TypeError("Population must be a SliceablePopulation instance")
        if len(pop) % num_islands != 0:
            raise ValueError(
                f"Population size {len(pop)} must be divisible by number of islands {num_islands}"
            )

        population_size_per_island = int(np.ceil(len(pop) / num_islands))
        total_evaluations = (
            population_size_per_island
            * generations_per_migration_event
            * num_islands
            * (migration_events + 1)
        )
        self.logger.info("Configuration:")
        self.logger.info(f"  Islands: {num_islands}")
        self.logger.info(
            f"  Generations per migration event: {generations_per_migration_event}"
        )
        self.logger.info(f"  Migration events: {migration_events}")
        self.logger.info(
            f"  Total generations: {generations_per_migration_event * (migration_events + 1)}"
        )
        self.logger.info(f"  Population per island: {population_size_per_island}")
        self.logger.info(f"  Total evaluations: ~{total_evaluations}")

        # Create PyGMO archipelago
        self.archipelago = pg.archipelago(
            t=topology or pg.topology(pg.fully_connected())
        )
        
        algo_name = pg.algorithm(self.algo_type()).get_name()

        for i in range(num_islands):
            island_record = Island(
                archipelago_id=self.archipelago_record.id,
                island_id=i,
                initial_population_size=population_size_per_island,
                algorithm_type=algo_name ,
                algorithm_parameters=algo_parameters or {},
                status=STATUS.INITIALIZED.value,
            )
            island_record.save_to_db()
            algo = pg.algorithm(
                TrackedAlgorithm(
                    island_id=i,
                    algo=self.algo_type,
                    gen=generations_per_migration_event,
                    seed=seed,
                    # cr=0.95,  # JLeitner (2010)
                    # m=0.01,  # JLeitner (2010)
                    **algo_parameters if algo_parameters else {},
                )
            )
            island = pg.island(
                algo=algo,
                pop=pop[
                    i * population_size_per_island : (i + 1)
                    * population_size_per_island
                ],
            )
            self.push_back(island)

    def push_back(self, island: pg.island):
        """Add an island to the archipelago."""
        self.archipelago.push_back(island)

    def evolve(self, n: int = None) -> None:
        """
        Evolve archipelago for n migration cycles with basic database tracking.
        Updates shared state for cross-process communication with TrackedAlgorithm.
        """
        self.logger.info(f"Starting evolution for {n} migration cycles")
        if n is None:
            n = self.migration_events
        try:
            for migration_cycle in range(n):
                self.logger.info(f"Migration cycle {migration_cycle + 1}/{n}")

                # Evolve once (one migration cycle)
                self.archipelago.evolve(1)
                self.archipelago.wait_check()

            self.logger.info("Evolution completed successfully")

        except Exception as e:
            self.logger.error(f"Evolution failed: {e}")
            raise

    def wait_check(self):
        """Wait for archipelago completion."""
        return self.archipelago.wait_check()

    def get_migration_log(self):
        """Get migration log from PyGMO."""
        return self.archipelago.get_migration_log()

    def __len__(self):
        """Number of islands."""
        return len(self.archipelago)

    def __getitem__(self, index):
        """Get island by index."""
        return self.archipelago[index]

    def __iter__(self):
        """Iterate over islands."""
        return iter(self.archipelago)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    problem = pg.problem(pg.rosenbrock(2))
    pop = SliceablePopulation(problem, 100)


    PyGMOBase.create_tables()
    archipelago = TrackedArchipelago(
        num_islands=4,
        pop=pop,
        algo=pg.sga,
        generations_per_migration_event=10,
        migration_events=5,
        seed=42,
    )

    # archipelago.evolve(3)
    # print("Migration log:", archipelago.get_migration_log())
