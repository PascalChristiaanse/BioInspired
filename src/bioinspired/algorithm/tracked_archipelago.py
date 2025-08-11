"""
Simplified PyGMO archipelago with database tracking.
Minimal implementation focusing on basic database operations.
Includes minimal multiprocessing communication for cross-process algorithm tracking.
"""

import time
import logging
import numpy as np
import multiprocessing as mp
from datetime import datetime
from typing import Optional, Dict

import pygmo as pg
from bioinspired.data.database import get_session_context
from bioinspired.data.pygmo_models import (
    Archipelago,
    Island,
    Individual,
    STATUS,
    PyGMOBase,
)
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
        topology: pg.topology = None,
    ):
        """
        Initialize tracked archipelago.

        Args:
            num_islands: Number of islands in the archipelago
            pop: Initial population as a SliceablePopulation
            algo: PyGMO algorithm to use for evolution
            algo_parameters: Parameters for the algorithm
            generations_per_migration_event: Generations before migration occurs
            migration_events: Number of migration events to perform
            topology: PyGMO topology (defaults to fully connected)
        """
        # Set a logger for this class
        self.logger = logging.getLogger(__class__.__name__)
        self.logger.info("Initializing TrackedArchipelago")

        self.num_islands = num_islands
        self.algo_type = algo
        self.migration_events = migration_events
        self.seed = pop.get_seed()
        self.archipelago_record: Archipelago = Archipelago(
            name="Tracked Archipelago",
            problem_class=pop.problem.get_name(),
            num_islands=num_islands,
            population_per_island=len(pop) / num_islands,
            generations_per_migration_event=generations_per_migration_event,
            migration_events=migration_events,
            total_generations=generations_per_migration_event * (migration_events + 1),
            seed=self.seed,
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

        # Store island records for later reference
        self.island_records = []

        for i in range(num_islands):
            # Store algorithm parameters including generations_per_migration_event and seed
            full_algo_parameters = {
                "generations_per_migration_event": generations_per_migration_event,
                "seed": self.seed,
                **(algo_parameters or {}),
            }

            island_record = Island(
                archipelago_id=self.archipelago_record.id,
                island_id=i,
                initial_population_size=population_size_per_island,
                algorithm_type=algo_name,
                algorithm_parameters=full_algo_parameters,
                status=STATUS.INITIALIZED.value,
            )
            island_record.save_to_db()

            # Store the island record for later use
            self.island_records.append(island_record)
            algo = pg.algorithm(
                TrackedAlgorithm(
                    island_id=island_record.id,
                    algo=self.algo_type,
                    archipelago_id=self.archipelago_record.id,
                    gen=generations_per_migration_event,
                    generations_per_migration_event=generations_per_migration_event,
                    current_migration_cycle=0,
                    seed=self.seed,
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
        """
        if n is None:
            n = self.migration_events
        self.logger.info(f"Starting evolution for {n} migration cycles")

        # Record start time and update archipelago record
        start_time = time.time()
        self.archipelago_record.update_in_db(
            status=STATUS.RUNNING.value, started_at=datetime.now()
        )

        try:
            for migration_cycle in range(0, n+1):
                self.logger.info(f"Migration cycle {migration_cycle}/{n}")

                # Update current migration event in archipelago record
                self.archipelago_record.update_in_db(
                    current_migration_event=migration_cycle
                )

                # Update island status to RUNNING
                for island_record in self.island_records:
                    island_record.update_in_db(status=STATUS.RUNNING.value)

                # Update all islands with new algorithm instances that have the correct migration cycle
                for i, island in enumerate(self.archipelago):
                    # Get the corresponding island record
                    island_record = self.island_records[i]

                    # Create new algorithm with updated migration cycle
                    new_algo = pg.algorithm(
                        TrackedAlgorithm(
                            island_id=island_record.id,
                            algo=self.algo_type,
                            archipelago_id=self.archipelago_record.id,
                            gen=island_record.algorithm_parameters.get(
                                "generations_per_migration_event", 5
                            ),
                            generations_per_migration_event=island_record.algorithm_parameters.get(
                                "generations_per_migration_event", 5
                            ),
                            current_migration_cycle=migration_cycle,
                            seed=island_record.algorithm_parameters.get("seed", 42),
                            **{
                                k: v
                                for k, v in island_record.algorithm_parameters.items()
                                if k not in ["generations_per_migration_event", "seed"]
                            },
                        )
                    )

                    # Set the new algorithm on the island
                    island.set_algorithm(new_algo)

                # Evolve once (one migration cycle)
                self.archipelago.evolve(1)
                self.archipelago.wait_check()

                # Update island fitness statistics after evolution
                for i, island in enumerate(self.archipelago):
                    try:
                        island_record = self.island_records[i]
                        population = island.get_population()

                        # Get current fitness statistics and convert to lists
                        current_champion_fitness = population.champion_f.tolist()[0]
                        current_worst_fitness = max(population.get_f().tolist())[0]

                        # Get existing values from database
                        existing_record = Island.get_by_id(island_record.id)
                        if existing_record:
                            # Determine best and worst fitness across all evolution cycles
                            best_fitness = current_champion_fitness
                            if existing_record.best_fitness is not None:
                                best_fitness = min(
                                    float(existing_record.best_fitness),
                                    current_champion_fitness,
                                )

                            worst_fitness = current_worst_fitness
                            if existing_record.worst_fitness is not None:
                                worst_fitness = max(
                                    float(existing_record.worst_fitness),
                                    current_worst_fitness,
                                )

                            # Update the island record directly using update_in_db
                            island_record.update_in_db(
                                best_fitness=float(best_fitness),
                                worst_fitness=float(worst_fitness),
                                generations_completed=island_record.generations_completed
                                + self.archipelago_record.generations_per_migration_event,
                            )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to update island {i} fitness statistics: {e}"
                        )
                        # Update with error message if fitness tracking fails
                        try:
                            island_record.update_in_db(error_message=str(e))
                        except Exception:
                            pass

            # Update island status to COMPLETED
            for island_record in self.island_records:
                island_record.update_in_db(status=STATUS.COMPLETED.value)

            # Calculate completion statistics for archipelago
            end_time = time.time()
            total_runtime = end_time - start_time

            # Gather fitness statistics from all islands
            all_fitness_values = np.array([], dtype=float)
            total_evaluations = 0

            for i, island in enumerate(self.archipelago):
                try:
                    population = island.get_population()
                    fitness_values = np.array(population.get_f().tolist()).flatten()
                    all_fitness_values = np.append(all_fitness_values, fitness_values)

                    # Count evaluations (population size * generations per island)
                    island_record = self.island_records[i]
                    total_evaluations += len(fitness_values) * (
                        island_record.generations_completed or 0
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to gather statistics from island {i}: {e}"
                    )

            # Calculate average fitness
            average_fitness = np.average(all_fitness_values)
            best_overall_fitness = np.min(all_fitness_values)
            worst_overall_fitness = np.max(all_fitness_values)
            # Update archipelago record with final statistics
            self.archipelago_record.update_in_db(
                status=STATUS.COMPLETED.value,
                completed_at=datetime.now(),
                total_runtime_seconds=float(total_runtime),
                best_fitness=float(best_overall_fitness),
                worst_fitness=float(worst_overall_fitness),
                average_fitness=float(average_fitness),
                total_evaluations=int(total_evaluations),
                current_migration_event=n,  # Final migration event number
            )

            self.logger.info("Evolution completed successfully")

        except Exception as e:
            # Calculate runtime even on failure
            end_time = time.time()
            total_runtime = end_time - start_time

            # Update archipelago record with failure information
            self.archipelago_record.update_in_db(
                status=STATUS.FAILED.value,
                completed_at=datetime.now(),
                total_runtime_seconds=float(total_runtime),
                exception_message=str(e),
            )

            # Update island status to FAILED on error with error message
            for island_record in self.island_records:
                island_record.update_in_db(
                    status=STATUS.FAILED.value, error_message=str(e)
                )
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
    problem = pg.problem(pg.rosenbrock(20))
    pop = SliceablePopulation(problem, 100)

    PyGMOBase.create_tables()
    archipelago = TrackedArchipelago(
        num_islands=4,
        pop=pop,
        algo=pg.sga,
        algo_parameters={"cr": 0.95, "m": 0.01},  # JLeitner (2010)
        generations_per_migration_event=10,
        migration_events=1,
        seed=42,
    )
    archipelago.evolve()
    # print("Migration log:", archipelago.get_migration_log())
