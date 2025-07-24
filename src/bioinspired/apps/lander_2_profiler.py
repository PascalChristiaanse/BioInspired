"""BioInspired - Lander 2 Profiler
Profile simulation setup time vs runtime for optimization purposes.
"""

import numpy as np
import time
from datetime import datetime

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries

# Import the bioinspired modules
from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.spacecraft import Lander2
from bioinspired.controllers import ConstantController


def create_initial_state():
    """Create the initial state for the spacecraft."""
    initial_state = np.array([0, 0, 0, 0, 0, 0])  # Initial position and velocity
    
    # Set initial rotation matrix (identity matrix)
    initial_rotation_matrix = np.eye(3)
    initial_rotational_velocity = np.array([0.0, 0, 0])  # Initial angular velocity
    
    # Set initial orientation by converting rotation matrix to quaternion
    initial_state_rotation = rotation_matrix_to_quaternion_entries(initial_rotation_matrix)
    
    # Complete initial state
    initial_state = np.concatenate(
        (initial_state, initial_state_rotation, initial_rotational_velocity)
    )
    
    return initial_state


def profile_simulation_setup():
    """Profile the time taken to set up a simulation."""
    print("Profiling simulation setup...")
    
    setup_start = time.perf_counter()
    
    # Create simulation
    simulator = EmptyUniverseSimulator()
    
    # Create initial state
    initial_state = create_initial_state()
    
    # Create controller
    controller = ConstantController(
        simulator=simulator,
        lander_name="Lander 2",
        target_name="Lander 2",
    )
    
    # Create spacecraft
    spacecraft = Lander2(
        initial_state=initial_state, 
        simulation=simulator, 
        controller=controller
    )
    
    setup_end = time.perf_counter()
    setup_time = setup_end - setup_start
    
    print(f"Setup completed in: {setup_time:.4f} seconds")
    
    return simulator, spacecraft, setup_time


def profile_simulation_runtime(simulator, simulation_time=100):
    """Profile the time taken to run a simulation."""
    print(f"Profiling simulation runtime (simulation_time={simulation_time}s)...")
    
    runtime_start = time.perf_counter()
    
    # Run simulation
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=simulation_time)
    
    runtime_end = time.perf_counter()
    runtime = runtime_end - runtime_start
    
    print(f"Simulation completed in: {runtime:.4f} seconds")
    print(f"Real-time factor: {simulation_time / runtime:.2f}x")
    
    return dynamics_simulator, runtime


def profile_multiple_runs(num_runs=5, simulation_time=100):
    """Profile multiple simulation runs to get average timings."""
    print(f"\nProfiling {num_runs} runs...")
    print("=" * 60)
    
    setup_times = []
    runtime_times = []
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        print("-" * 30)
        
        # Profile setup
        simulator, spacecraft, setup_time = profile_simulation_setup()
        setup_times.append(setup_time)
        
        # Profile runtime
        dynamics_simulator, runtime = profile_simulation_runtime(simulator, simulation_time)
        runtime_times.append(runtime)
        
        # Clean up
        del simulator, spacecraft, dynamics_simulator
    
    # Calculate statistics
    avg_setup = np.mean(setup_times)
    std_setup = np.std(setup_times)
    avg_runtime = np.mean(runtime_times)
    std_runtime = np.std(runtime_times)
    
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    print(f"Setup Time Statistics ({num_runs} runs):")
    print(f"  Average: {avg_setup:.4f} ± {std_setup:.4f} seconds")
    print(f"  Min:     {min(setup_times):.4f} seconds")
    print(f"  Max:     {max(setup_times):.4f} seconds")
    print()
    print(f"Runtime Statistics ({num_runs} runs, {simulation_time}s sim time):")
    print(f"  Average: {avg_runtime:.4f} ± {std_runtime:.4f} seconds")
    print(f"  Min:     {min(runtime_times):.4f} seconds")
    print(f"  Max:     {max(runtime_times):.4f} seconds")
    print(f"  Avg Real-time factor: {simulation_time / avg_runtime:.2f}x")
    print()
    print(f"Setup vs Runtime ratio: {avg_setup / avg_runtime:.2f}")
    print(f"Total time per run: {avg_setup + avg_runtime:.4f} seconds")
    
    return {
        'setup_times': setup_times,
        'runtime_times': runtime_times,
        'avg_setup': avg_setup,
        'avg_runtime': avg_runtime,
        'std_setup': std_setup,
        'std_runtime': std_runtime
    }


def profile_different_simulation_times():
    """Profile how runtime scales with simulation time."""
    print("\nProfiling different simulation times...")
    print("=" * 60)
    
    sim_times = [10, 25, 50, 100, 200]
    results = {}
    
    for sim_time in sim_times:
        print(f"\nTesting simulation time: {sim_time}s")
        print("-" * 30)
        
        # Just do setup once for this test
        simulator, spacecraft, setup_time = profile_simulation_setup()
        
        # Profile runtime
        dynamics_simulator, runtime = profile_simulation_runtime(simulator, sim_time)
        
        results[sim_time] = {
            'setup_time': setup_time,
            'runtime': runtime,
            'real_time_factor': sim_time / runtime
        }
        
        # Clean up
        del simulator, spacecraft, dynamics_simulator
    
    print("\n" + "=" * 60)
    print("SIMULATION TIME SCALING")
    print("=" * 60)
    print(f"{'Sim Time (s)':<12} {'Runtime (s)':<12} {'Real-time Factor':<15} {'Setup (s)':<10}")
    print("-" * 55)
    
    for sim_time, data in results.items():
        print(f"{sim_time:<12} {data['runtime']:<12.4f} {data['real_time_factor']:<15.2f} {data['setup_time']:<10.4f}")
    
    return results


def main():
    print("=" * 60)
    print("BioInspired - Lander 2 Performance Profiler")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Profile multiple runs with standard settings
    standard_results = profile_multiple_runs(num_runs=5, simulation_time=100)
    
    # Profile different simulation times
    scaling_results = profile_different_simulation_times()
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Recommendations based on results
    setup_time = standard_results['avg_setup']
    runtime = standard_results['avg_runtime']
    
    print("\nRECOMMENDations:")
    print("-" * 20)
    
    if setup_time > runtime:
        print("⚠️  Setup time is longer than runtime!")
        print("   Consider optimizing spacecraft initialization for multiple runs.")
    else:
        print("✅ Runtime dominates over setup time.")
    
    if runtime < 1.0:
        print("✅ Fast simulation - good for parameter sweeps.")
    elif runtime < 5.0:
        print("⚠️  Moderate simulation time - consider optimization for large parameter sweeps.")
    else:
        print("🔥 Slow simulation - optimization highly recommended for multiple runs.")
    
    ratio = setup_time / runtime
    if ratio > 0.5:
        print(f"⚠️  Setup overhead is {ratio:.1f}x runtime - consider reusing spacecraft objects.")
    
    return standard_results, scaling_results


if __name__ == "__main__":
    main()
