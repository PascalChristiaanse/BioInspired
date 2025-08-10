"""Tracked Algorithm Module"""

import pygmo as pg
from typing import Union, Callable, Dict, Any, Type, Optional
import logging


logger = logging.getLogger(__name__)


class TrackedAlgorithm:
    """
    A serializable wrapper for PyGMO algorithms that provides telemetry and logging capabilities.

    This class can wrap any PyGMO algorithm to provide:
    - Generation-by-generation tracking
    - Population logging and saving
    - Cross-process communication with TrackedArchipelago via multiprocessing manager
    
    Designed to be pickle-friendly for multiprocessing.
    """

    def __init__(
        self, 
        algo: Union[Type, Callable], 
        island_id: int = None, 
        gen: int = 1,
        session_factory: Optional[Callable] = None,
        archipelago_id: Optional[int] = None,
        shared_state: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the TrackedAlgorithm.

        Args:
            algo: PyGMO algorithm class (e.g., pg.sga, pg.de, pg.pso)
            island_id: ID of the island this algorithm is running on
            gen: Number of generations to run
            session_factory: SQLAlchemy session factory for database access
            archipelago_id: ID of the archipelago this algorithm belongs to
            shared_state: Multiprocessing manager dict for cross-process communication
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
        # Store these as simple types to avoid pickle issues
        self.session_factory = session_factory
        self.archipelago_id = archipelago_id
        self.shared_state = shared_state
        self.kwargs = kwargs
        
        logger.info(f"TrackedAlgorithm initialized for island {island_id}, archipelago {archipelago_id}")

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
        
        for gen in range(self.generations):
            # Evolve one generation
            current_pop = algorithm.evolve(current_pop)
            
            # Save champion with migration context from shared state
            self._save_champion(current_pop, gen, island_id=self.island_id)
            
        return current_pop

    def _save_champion(self, pop: pg.population, generation_number: int, island_id: int = None):
        """
        Save the best individual from the population as a champion.
        Uses shared state for cross-process communication to get migration context.

        Args:
            pop: PyGMO population to extract the champion from
            generation_number: Current generation number within this evolution cycle
            island_id: ID of the island
        """
        # Get current migration event from shared state (multiprocessing manager)
        current_migration_event = 0
        if self.shared_state and 'current_migration_event' in self.shared_state:
            current_migration_event = self.shared_state['current_migration_event']
        
        champion_x = pop.champion_x
        champion_f = pop.champion_f[0] if len(pop.champion_f) > 0 else pop.champion_f
        
        # Save to database if session factory is available
        if self.session_factory and self.archipelago_id:
            try:
                # Import only when needed to avoid pickle issues
                from bioinspired.data.pygmo_models import Individual, Island
                
                with self.session_factory() as session:
                    # Get island record
                    island = session.query(Island).filter(
                        Island.archipelago_id == self.archipelago_id,
                        Island.island_id == island_id
                    ).first()
                    
                    if not island:
                        logger.warning(f"Island {island_id} not found in database")
                        return
                    
                    # Create individual record for the champion
                    individual = Individual(
                        archipelago_id=self.archipelago_id,
                        island_id=island.id,
                        individual_ID=0,  # Champion is always ID 0 for our purposes
                        generation_number=current_migration_event,  # Use migration event as generation number
                        chromosome=champion_x.tolist() if hasattr(champion_x, 'tolist') else list(champion_x),
                        chromosome_size=len(champion_x),
                        fitness=float(champion_f),
                        is_champion=True,
                        additional_metrics={
                            'local_generation': generation_number,  # Track local generation within migration cycle
                            'migration_event': current_migration_event
                        }
                    )
                    
                    session.add(individual)
                    session.commit()
                    
                    logger.info(f"Saved champion: fitness={champion_f:.6f}, migration_event={current_migration_event}, local_gen={generation_number}, island={island_id}")
                    
            except Exception as e:
                logger.error(f"Failed to save champion to database: {e}")
        else:
            # Fallback to simple logging if no database access
            logger.info(f"Champion fitness: {champion_f:.6f} at generation {generation_number} (migration_event={current_migration_event}) for island {island_id}")

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
            "archipelago_id": self.archipelago_id
        }
