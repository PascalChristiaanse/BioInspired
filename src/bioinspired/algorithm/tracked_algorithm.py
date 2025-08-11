"""Tracked Algorithm Module"""

import pygmo as pg
from typing import Union, Callable, Dict, Any, Type, Optional
import logging
from bioinspired.data import Individual, Island


logger = logging.getLogger(__name__)


class TrackedAlgorithm:
    """
    A serializable wrapper for PyGMO algorithms that provides telemetry and logging capabilities.

    This class can wrap any PyGMO algorithm to provide:
    - Generation-by-generation tracking
    - Population logging and saving
    - Cross-process communication with TrackedArchipelago via class variables

    Designed to be pickle-friendly for multiprocessing.
    """
    
    # Class variable to store current migration cycle (shared across all instances)
    _current_migration_cycle = 0
    _generations_per_migration_event = 1

    def __init__(
        self,
        algo: Union[Type, Callable],
        island_id: int = None,
        gen: int = 1,
        archipelago_id: Optional[int] = None,
        generations_per_migration_event: int = 1,
        current_migration_cycle: int = 0,
        **kwargs,
    ):
        """
        Initialize the TrackedAlgorithm.

        Args:
            algo: PyGMO algorithm class (e.g., pg.sga, pg.de, pg.pso)
            island_id: ID of the island this algorithm is running on
            gen: Number of generations to run
            archipelago_id: ID of the archipelago this algorithm belongs to
            generations_per_migration_event: Number of generations per migration event
            current_migration_cycle: Current migration cycle number
            **kwargs: Algorithm parameters
        """

        # Check if algorithm_type is a class or callable, otherwise raise ValueError
        if not (callable(algo) or isinstance(algo, type)):
            raise ValueError(
                "algorithm_type must be a class or callable, got: {}".format(type(algo))
            )

        self.island_id = island_id
        self.algo = algo
        self.generations = gen
        self.archipelago_id = archipelago_id
        self.generations_per_migration_event = generations_per_migration_event
        self.current_migration_cycle = current_migration_cycle
        self.kwargs = kwargs

        # Set class variable if provided
        if generations_per_migration_event:
            TrackedAlgorithm._generations_per_migration_event = generations_per_migration_event

        # logger.info(
        #     f"TrackedAlgorithm initialized for island {island_id}, archipelago {archipelago_id}"
        # )

    def evolve(self, pop: pg.population) -> pg.population:
        """
        Evolve the population for the specified number of generations.

        Args:
            pop: PyGMO population to evolve

        Returns:
            Evolved population
        """
        algorithm = pg.algorithm(self.algo(gen=1, **self.kwargs))
        current_pop = pop
        island_record = Island.get_by_id(self.island_id)
        for gen in range(self.generations):
            # Evolve one generation
            current_pop = algorithm.evolve(current_pop)
            island_record.generations_completed += 1
            island_record.update_in_db()
            
            # Save champion
            champion = Individual(
                individual_ID=current_pop.best_idx(),
                island_id=self.island_id,
                archipelago_id=self.archipelago_id,
                generation_number=gen + 1,
                migration_event=self.current_migration_cycle,
                total_generation=(self.current_migration_cycle * self.generations_per_migration_event) + (gen + 1),
                chromosome=current_pop.champion_x.tolist(),
                chromosome_size=len(current_pop.champion_x.tolist()),
                fitness=current_pop.champion_f.tolist()[0],
                is_champion=True,
            )
            champion.save_to_db()
        return current_pop

    def set_migration_cycle(self, migration_cycle: int) -> None:
        """Update the current migration cycle."""
        TrackedAlgorithm._current_migration_cycle = migration_cycle

    def get_name(self) -> str:
        """Get a descriptive name for this algorithm."""
        algo_name = str(self.algo).split(".")[-1].replace("'>", "").upper()
        return f"Tracked {algo_name} ({self.generations} generations)"

    def get_log(self) -> Dict[str, Any]:
        """Get algorithm information for logging."""
        return {
            "algorithm_name": self.get_name(),
            "generations": self.generations,
            "island_id": self.island_id,
            "archipelago_id": self.archipelago_id,
        }
