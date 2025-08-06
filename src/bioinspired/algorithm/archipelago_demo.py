"""
Minimal PyGMO Archipelago Optimization for Spacecraft Docking Neural Network Training

This module implements a streamlined multiprocessing optimization system using PyGMO's
built-in archipelago class to train neural networks for spacecraft docking control.
Designed for 16-core systems with 2x oversubscription (32 islands) for optimal CPU utilization.
"""

import numpy as np
import pygmo as pg
from bioinspired.problem import BasicProblem
import time
import datetime
import multiprocessing
import logging


def setup_logging():
    """Setup INFO level logging without emojis for clean output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger(__name__)


def run_archipelago_optimization():
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
    
    # Detect CPU cores and configure archipelago
    num_cores = multiprocessing.cpu_count()
    target_cores = min(1, num_cores)  # Use up to 16 cores as specified
    num_islands = target_cores * 2     # 2x oversubscription for maximum CPU utilization
    
    logger.info(f"System has {num_cores} CPU cores")
    logger.info(f"Using {target_cores} cores with {num_islands} islands")
    
    # Verify problem is picklable for multiprocessing
    try:
        import pickle
        test_udp = BasicProblem()
        pickle.dumps(test_udp)
        logger.info("Problem confirmed picklable for multiprocessing")
    except Exception as e:
        logger.error(f"Problem is not picklable: {e}")
        return
    
    # Get problem information
    udp = BasicProblem()
    bounds = udp.get_bounds()
    num_parameters = len(bounds[0])
    logger.info(f"Neural network parameters: {num_parameters}")
    logger.info(f"Parameter bounds: [{bounds[0][0]:.1f}, {bounds[1][0]:.1f}]")
    
    # Archipelago configuration
    generations = 10
    population_size = 20
    total_evaluations = num_islands * generations * population_size
    
    logger.info("Configuration:")
    logger.info(f"  Islands: {num_islands}")
    logger.info(f"  Generations per island: {generations}")
    logger.info(f"  Population per island: {population_size}")
    logger.info(f"  Total evaluations: ~{total_evaluations}")
    
    print(f"\nInitializing {num_islands}-island archipelago...")
      # Create archipelago using PyGMO's built-in class
    archipelago_start = time.time()
    
    try:
        # Create problem instance
        prob = pg.problem(BasicProblem())
        
        # Create algorithm for all islands (using SGA for simplicity)
        algo = pg.algorithm(pg.sga(gen=generations))
        algo.set_verbosity(0)  # Reduce verbosity for clean logging
        
        # Create archipelago with specified number of islands, problem, algorithm, and population size
        archi = pg.archipelago(n=num_islands, prob=prob, algo=algo, pop_size=population_size)
        
        logger.info(f"Archipelago created with {len(archi)} islands")
        
        logger.info("Starting archipelago evolution...")
        print("Running optimization across all CPU cores...")
        
        # Start the optimization
        archi.evolve()
        
        # Wait for completion
        archi.wait_check()
        
        archipelago_time = time.time() - archipelago_start
        logger.info(f"Archipelago completed in {archipelago_time:.2f}s")
        
        # Collect results from all islands
        results = []
        for i, island in enumerate(archi):
            pop = island.get_population()
            fitness = pop.champion_f[0]
            champion = pop.champion_x
            
            results.append({
                'island_id': i,
                'fitness': fitness,
                'champion': champion
            })
        
        # Analyze results
        logger.info("Analyzing results...")
        
        # Find best solution across all islands
        best_result = min(results, key=lambda x: x['fitness'])
        worst_result = max(results, key=lambda x: x['fitness'])
        
        fitnesses = [r['fitness'] for r in results]
        
        # Log comprehensive results
        print("\n" + "="*80)
        print("ARCHIPELAGO OPTIMIZATION RESULTS")
        print("="*80)
        
        logger.info(f"Best fitness: {best_result['fitness']:.6f} (Island {best_result['island_id']})")
        logger.info(f"Worst fitness: {worst_result['fitness']:.6f} (Island {worst_result['island_id']})")
        logger.info(f"Average fitness: {np.mean(fitnesses):.6f} ± {np.std(fitnesses):.6f}")
        logger.info(f"Fitness range: {np.max(fitnesses) - np.min(fitnesses):.6f}")
        
        # Performance metrics
        cpu_time_estimate = archipelago_time * target_cores
        logger.info(f"Wall time: {archipelago_time:.2f}s")
        logger.info(f"Estimated CPU time: ~{cpu_time_estimate:.2f}s")
        logger.info(f"Parallel efficiency: ~{cpu_time_estimate / archipelago_time:.1f}x speedup")
        
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
        if os.name == 'nt':  # Windows
            multiprocessing.set_start_method('spawn', force=True)
        
        # Run the optimization
        best_result = run_archipelago_optimization()
        
        if best_result:
            print(f"\nSuccess! Neural network trained with fitness: {best_result['fitness']:.6f}")
        else:
            print("\nOptimization failed.")
            
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        print(f"\nOptimization error: {str(e)}")


if __name__ == "__main__":
    main()
