"""Evolutionary Algorithm
This module implements pygmo's optimization algorithms for benchmarking and experimenting with multithreaded archipelagos.
"""

import pygmo as pg
import time
from tqdm import tqdm
import threading


class ToyProblem:
    """A moderately computationally intensive constrained optimization problem for testing."""
    
    def __init__(self, dim):
        self.dim = dim
    
    def fitness(self, x):
        """Objective function and constraints with moderate computation."""
        # Moderately expensive objective function
        obj = 0.0
        
        # Some computational complexity but not excessive
        for i in range(self.dim):
            obj += x[i] ** 2 + 0.1 * x[i] ** 4
            if i < self.dim - 1:
                obj += (x[i] - x[i+1]) ** 2
        
        # Add some nonlinearity
        obj += sum(abs(x[i]) ** 1.5 for i in range(self.dim))
        
        # Constraint computations
        # Constraint 1: sum(x) >= 1
        c1 = 1 - sum(x)
        
        # Constraint 2: Nonlinear constraint
        c2 = sum(x[i] ** 2 for i in range(min(5, self.dim))) - 2.0
        
        return [obj, c1, c2]
    
    def get_bounds(self):
        """Variable bounds."""
        return ([-5] * self.dim, [5] * self.dim)
    
    def get_nec(self):
        """Number of equality constraints."""
        return 0
    
    def get_nic(self):
        """Number of inequality constraints."""
        return 2


def monitor_archipelago_progress(archi, interval=5, stop_event=None):
    """Monitor and display progress of archipelago evolution."""
    import time
    
    start_time = time.time()
    while not (stop_event and stop_event.is_set()):
        elapsed = time.time() - start_time
        
        # Get current best fitness across all islands
        try:
            current_best = float('inf')
            active_islands = 0
            
            for island in archi:
                try:
                    pop = island.get_population()
                    if len(pop.get_f()) > 0:
                        island_best = pop.champion_f[0]
                        if island_best < current_best:
                            current_best = island_best
                        active_islands += 1
                except Exception:
                    # Island might not be ready yet
                    pass
            
            if current_best != float('inf'):
                print(f"\rProgress: {elapsed:.1f}s | Active islands: {active_islands}/{len(archi)} | Best fitness: {current_best:.6f}", end="", flush=True)
            else:
                print(f"\rProgress: {elapsed:.1f}s | Islands initializing...", end="", flush=True)
        except Exception:
            print(f"\rProgress: {elapsed:.1f}s | Islands evolving...", end="", flush=True)
        
        time.sleep(interval)
    
    print()  # New line after progress


def benchmark_single_threaded():
    """Benchmark single-threaded optimization."""
    print("=" * 60)
    print("SINGLE-THREADED BENCHMARK")
    print("=" * 60)
    
    # Create moderately sized problem
    prob = pg.problem(ToyProblem(80))  # Reduced from 200 to 80
    prob.c_tol = [1e-4, 1e-4]  # Back to 2 constraints
    
    algo = pg.algorithm(pg.cstrs_self_adaptive(iters=800))  # Reduced from 2000 to 800
    algo.set_verbosity(100)  # More frequent updates
    
    # Moderate population size
    pop = pg.population(prob, 100)  # Reduced from 150 to 100
    
    print("Starting single-threaded evolution...")
    start_time = time.time()
    pop = algo.evolve(pop)
    end_time = time.time()
    
    print(f"Single-threaded time: {end_time - start_time:.2f} seconds")
    print(f"Best fitness: {pop.champion_f}")
    print(f"Feasible solutions: {len([f for f in pop.get_f() if prob.feasibility_f(f)])}")
    
    return end_time - start_time, pop.champion_f


def benchmark_archipelago():
    """Benchmark multithreaded archipelago optimization."""
    print("\n" + "=" * 60)
    print("MULTITHREADED ARCHIPELAGO BENCHMARK")
    print("=" * 60)
    
    # Create algorithm and problem - moderate size
    algo = pg.algorithm(pg.cstrs_self_adaptive(iters=800))  # Reduced from 2000 to 800
    prob = pg.problem(ToyProblem(80))  # Reduced from 200 to 80
    prob.c_tol = [1e-4, 1e-4]  # Back to 2 constraints
    
    # Create archipelago with 16 islands (matching your 16-core CPU)
    archi = pg.archipelago(n=16, algo=algo, prob=prob, pop_size=100)  # Reduced pop size
    
    print("Archipelago created:")
    print(archi)
    print("Starting archipelago evolution...")
    
    # Run evolution with progress monitoring
    start_time = time.time()
    archi.evolve()
    
    # Start progress monitor in background with stop event
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_archipelago_progress, args=(archi, 2, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Monitor progress
    print("Waiting for evolution to complete...")
    archi.wait_check()  # Wait for all islands to finish
    stop_event.set()  # Signal monitor to stop
    monitor_thread.join(timeout=1)  # Wait for monitor to finish
    end_time = time.time()
    
    print(f"Archipelago time: {end_time - start_time:.2f} seconds")
    
    # Get best solution across all islands
    best_fitness = float('inf')
    best_island = 0
    
    for i, island in enumerate(archi):
        island_best = island.get_population().champion_f
        if island_best[0] < best_fitness:
            best_fitness = island_best[0]
            best_island = i
    
    print(f"Best fitness across all islands: {best_fitness}")
    print(f"Best found on island: {best_island}")
    print(f"Total islands: {len(archi)}")
    
    return end_time - start_time, best_fitness


def compare_algorithms():
    """Compare different optimization algorithms using archipelagos."""
    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON")
    print("=" * 60)
    
    prob = pg.problem(ToyProblem(50))  # Reduced from 100 to 50
    prob.c_tol = [1e-4, 1e-4]  # Back to 2 constraints
    
    algorithms = [
        ("Constrained Self-Adaptive", pg.cstrs_self_adaptive(iters=400)),  # Reduced iterations
        ("Differential Evolution", pg.de(gen=400)),  # Reduced generations
        ("Particle Swarm", pg.pso(gen=400)),  # Reduced generations
        ("Simulated Annealing", pg.simulated_annealing(Ts=10, Tf=0.01)),
    ]
    
    results = {}
    
    for name, algo_udp in algorithms:
        print(f"\nTesting {name}...")
        
        algo = pg.algorithm(algo_udp)
        archi = pg.archipelago(n=8, algo=algo, prob=prob, pop_size=60)  # Reduced pop size
        
        start_time = time.time()
        archi.evolve()
        archi.wait_check()
        end_time = time.time()
        
        # Find best across islands
        best_fitness = min(island.get_population().champion_f[0] for island in archi)
        
        results[name] = {
            'time': end_time - start_time,
            'fitness': best_fitness
        }
        
        print(f"  Time: {results[name]['time']:.2f}s")
        print(f"  Best fitness: {results[name]['fitness']:.4f}")
    
    return results


def benchmark_schwefel():
    """Benchmark on Schwefel function (from original code)."""
    print("\n" + "=" * 60)
    print("SCHWEFEL FUNCTION BENCHMARK")
    print("=" * 60)
    
    # Create moderate Schwefel problem
    prob = pg.problem(pg.schwefel(dim=50))  # Reduced from 100 to 50
    
    algorithms = [
        ("DE", pg.de(gen=600)),  # Reduced generations
        ("PSO", pg.pso(gen=600)),  # Reduced generations
        ("Self-Adaptive DE", pg.sade(gen=600)),  # Reduced generations
    ]
    
    for name, algo_udp in algorithms:
        print(f"\nTesting {name} on Schwefel...")
        
        algo = pg.algorithm(algo_udp)
        archi = pg.archipelago(n=12, algo=algo, prob=prob, pop_size=50)  # Reduced pop size
        
        start_time = time.time()
        archi.evolve()
        archi.wait_check()
        end_time = time.time()
        
        # Find best across islands
        best_fitness = min(island.get_population().champion_f[0] for island in archi)
        optimal_schwefel = 418.9829 * 50  # Known optimum for 50D Schwefel
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Distance from optimum: {best_fitness - (-optimal_schwefel):.4f}")


def main():
    """Run all benchmarks."""
    print("PYGMO ARCHIPELAGO BENCHMARKING")
    print("Threading available:", pg.mp_island.get_pool_size())
    
    # Your original archipelago example - now moderately sized
    print("\n" + "=" * 60)
    print("YOUR ORIGINAL EXAMPLE (MODERATE SIZE)")
    print("=" * 60)
    
    a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=400))  # Further reduced
    p_toy = pg.problem(ToyProblem(50))  # Reduced from 80 to 50
    p_toy.c_tol = [1e-4, 1e-4]  # Back to 2 constraints
    
    archi = pg.archipelago(n=16, algo=a_cstrs_sa, prob=p_toy, pop_size=60)  # Reduced islands and pop
    print("Archipelago created:")
    print(archi)
    
    print("\nEvolving...")
    start_time = time.time()
    archi.evolve()
    
    # Use the progress monitor with stop event
    import threading
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=monitor_archipelago_progress, args=(archi, 2, stop_event))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    archi.wait_check()
    stop_event.set()  # Signal monitor to stop
    monitor_thread.join(timeout=1)  # Wait for monitor to finish
    end_time = time.time()
    
    best_fitness = min(island.get_population().champion_f[0] for island in archi)
    print(f"Evolution completed in {end_time - start_time:.2f} seconds")
    print(f"Best fitness found: {best_fitness:.6f}")
    
    # Run additional benchmarks
    try:
        single_time, single_fitness = benchmark_single_threaded()
        archi_time, archi_fitness = benchmark_archipelago()
        
        print(f"\nSpeedup: {single_time / archi_time:.2f}x")
        print(f"Fitness improvement: {(single_fitness - archi_fitness) / single_fitness * 100:.2f}%")
        
        compare_algorithms()
        benchmark_schwefel()
        
    except Exception as e:
        print(f"Error in additional benchmarks: {e}")


if __name__ == "__main__":
    main()