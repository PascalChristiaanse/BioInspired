"""
Minimal PyGMO Archipelago Optimization for Spacecraft Docking Neural Network Training

This module implements a streamlined multiprocessing optimization system using PyGMO's
built-in archipelago class to train neural networks for spacecraft docking control.
Designed for 16-core systems with 2x oversubscription (32 islands) for optimal CPU utilization.
"""

import numpy as np
import pygmo as pg

from bioinspired.algorithm import InitializerBFE, SliceablePopulation

from bioinspired.problem import StopNeuronBasicProblem as ProblemClass
import time
import datetime
import multiprocessing
import logging


class TelemetrySGA:
    def __init__(self, generations, *args, **kwargs):
        self.generations = generations
        self.args = args
        self.kwargs = kwargs

    def evolve(self, pop):
        new_pop = pop
        algo = pg.algorithm(pg.sga(*self.args, **self.kwargs))
        algo.set_verbosity(2)  # Reduce verbosity for clean logging
        for gen in range(self.generations):
            print(f"Generation {gen + 1}/{self.generations}")
            new_pop = algo.evolve(new_pop)
            # Log the current generation's best fitness
            best_fitness = pop.champion_f
            champion = pop.champion_x
            # Save champion to file using np.save
            np.save(f"champion_gen_{gen + 1}_time:{time.ctime()}.npy".replace(':', "-"), champion)
            print(f"Best fitness in generation {gen + 1}: {best_fitness[0]:.6f}")
        return new_pop

    def get_name(self):
        return f"Telemetry SGA ({self.generations} generations)"


def setup_logging():
    """Setup INFO level logging without emojis for clean output"""

    # disable numbas logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.getLogger("numba").setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def run_archipelago_optimization(seed=42):
    """
    Run multi-island optimization using PyGMO's built-in archipelago class

    Utilizes all 16 CPU cores with 2x oversubscription (32 islands) for maximum
    CPU utilization while training neural networks for spacecraft docking.
    """
    logger = setup_logging()

    print("=" * 80)
    print("PYGMO ARCHIPELAGO - BIOINSPIRED SPACECRAFT DOCKING NEURAL NETWORK")
    print("=" * 80)
    print(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Verify problem is picklable for multiprocessing
    try:
        import pickle

        test_udp = ProblemClass()
        pickle.dumps(test_udp)
        logger.info("Problem confirmed picklable for multiprocessing")
    except Exception as e:
        logger.error(f"Problem is not picklable: {e}")
        return

    # Detect CPU cores and configure archipelago
    num_cores = multiprocessing.cpu_count()
    target_cores = min(8, num_cores)
    num_islands = target_cores * 1  # 2x oversubscription for maximum CPU utilization

    logger.info(f"System has {num_cores} CPU cores")
    logger.info(f"Using {target_cores} cores with {num_islands} islands")

    # Archipelago configuration
    generations_per_migration_event = 3
    migration_events = 1
    population_size = 16 * num_islands  # Total population size across all islands
    population_size_per_island = int(np.ceil(population_size / num_islands))
    total_evaluations = (
        population_size_per_island
        * generations_per_migration_event
        * num_islands
        * (migration_events + 1)
    )

    logger.info("Configuration:")
    logger.info(f"  Islands: {num_islands}")
    logger.info(f"  Generations per migration event: {generations_per_migration_event}")
    logger.info(f"  Migration events: {migration_events}")
    logger.info(
        f"  Total generations: {generations_per_migration_event * (migration_events + 1)}"
    )
    logger.info(f"  Population per island: {population_size_per_island}")
    logger.info(f"  Total evaluations: ~{total_evaluations}")

    # Get problem information
    udp = ProblemClass()
    bounds = udp.get_bounds()
    num_parameters = len(bounds[0])
    logger.info(f"Neural network parameters: {num_parameters}")
    logger.info(f"Parameter bounds: [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}]")

    print(f"\nInitializing {num_islands}-island archipelago...")

    initial_population_time = time.time()
    # Create an initial population using the Multiprocessing BFE
    initial_population = SliceablePopulation(
        udp,
        size=population_size_per_island * num_islands,
        b=pg.mp_bfe(64),
        # b=InitializerBFE(
        #     ProblemClass().get_bounds()[0].shape[0], int(round(population_size / 64))
        # ),
        seed=seed,
    )
    initial_population_time = time.time() - initial_population_time
    print(f"Initial population created in {initial_population_time:.2f}s")

    # Create archipelago using PyGMO's built-in class
    archipelago_start = time.time()
    try:
        # Set up the algorithm for all islands
        algo = pg.algorithm(
            TelemetrySGA(
                generations=generations_per_migration_event,
                gen=generations_per_migration_event,
                cr=0.95,  # JLeitner (2010)
                m=0.01,  # JLeitner (2010)
                seed=seed,
            )
        )
        # algo = pg.algorithm(
        #     pg.sga(
        #         gen=generations_per_migration_event,
        #         cr=0.95,  # JLeitner (2010)
        #         m=0.01,  # JLeitner (2010)
        #         seed=seed,
        #     )
        # )
        # algo.set_verbosity(2)  # Reduce verbosity for clean logging

        # Create empty archipelago
        archi = pg.archipelago(
            t=pg.topology(pg.fully_connected()),
            # algo=algo,
        )
        for i in range(num_islands):
            island = pg.island(
                # udi=pg.mp_island(use_pool=True),
                pop=initial_population[
                    i * population_size_per_island : (i + 1)
                    * population_size_per_island
                ],
                algo=algo,
            )
            archi.push_back(island)

        print("Archipelago created with the following islands:")
        for i, island in enumerate(archi):
            print(f"  Island {i + 1}: {len(island.get_population())} individuals")

        print(archi)

        print("Running optimization across all CPU cores...")

        # Start the optimization
        test = archi.evolve(migration_events + 1)

        print(archi)
        # Wait for completion
        # wait 5 seconds
        print("Waiting for archipelago to complete...")
        time.sleep(5)  # Simulate some processing time
        archi.wait_check()
        print(test)
        # Present migration log
        print("\nMigration log:")
        print(archi.get_migration_log())

        archipelago_time = time.time() - archipelago_start
        logger.info(f"Archipelago completed in {archipelago_time:.2f}s")
        archi.evolve()
        print(archi)
        archi.wait_check()
        # Collect results from all islands
        results = []
        for i, island in enumerate(archi):
            pop = island.get_population()
            fitness = pop.champion_f[0]
            champion = pop.champion_x

            results.append({"island_id": i, "fitness": fitness, "champion": champion})

        # Analyze results
        logger.info("Analyzing results...")

        # Find best solution across all islands
        best_result = min(results, key=lambda x: x["fitness"])
        worst_result = max(results, key=lambda x: x["fitness"])

        fitnesses = [r["fitness"] for r in results]

        # Log comprehensive results
        print("\n" + "=" * 80)
        print("ARCHIPELAGO OPTIMIZATION RESULTS")
        print("=" * 80)

        logger.info(
            f"Best fitness: {best_result['fitness']:.6f} (Island {best_result['island_id']})"
        )
        logger.info(
            f"Worst fitness: {worst_result['fitness']:.6f} (Island {worst_result['island_id']})"
        )
        logger.info(
            f"Average fitness: {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}"
        )
        logger.info(f"Fitness range: {np.max(fitnesses) - np.min(fitnesses):.6f}")

        # Performance metrics
        cpu_time_estimate = archipelago_time * target_cores
        logger.info(f"Wall time: {archipelago_time:.2f}s")
        logger.info(f"Estimated CPU time: ~{cpu_time_estimate:.2f}s")
        logger.info(
            f"Parallel efficiency: ~{cpu_time_estimate / archipelago_time:.1f}x speedup"
        )

        # Neural network solution details
        logger.info("Best neural network solution:")
        logger.info(f"  Parameters optimized: {len(best_result['champion'])}")
        logger.info(f"  Final cost function value: {best_result['fitness']:.6f}")

        print("\nOptimization complete!")
        print(f"Best fitness achieved: {best_result['fitness']:.6f}")
        print(f"Total optimization time: {archipelago_time:.2f}s")
        print(f"CPU cores utilized: {target_cores}")
        print(f"Islands processed: {num_islands}")

        return best_result

    except Exception as e:
        logger.error(f"Archipelago optimization failed: {str(e)}")
        raise


def main():
    """Main entry point for the archipelago optimization"""
    try:
        # Ensure proper multiprocessing on Windows
        import os

        if os.name == "nt":  # Windows
            multiprocessing.set_start_method("spawn", force=True)

        # Run the optimization
        best_result = run_archipelago_optimization()

        if best_result:
            print(
                f"\nSuccess! Neural network trained with fitness: {best_result['fitness']:.6f}"
            )
        else:
            print("\nOptimization failed.")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"\nOptimization error: {str(e)}")


if __name__ == "__main__":
    main()
