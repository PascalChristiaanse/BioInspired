"""BioInspired - Timestep Convergence Analyzer
Analyze simulation accuracy vs timestep size to determine optimal timestep for desired precision.

Enhanced with comprehensive angular analysis including:
- Angular rate error (magnitude difference of angular velocity vectors)
- Orientation error (angular separation between quaternions)
- Enhanced visualization with 4 angular metrics in rotational error plots
- Detailed angular accuracy recommendations for all error methods
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.numerical_simulation.propagation_setup import integrator

# Import the bioinspired modules
from bioinspired.simulation import EmptyUniverseSimulatorAdjustable
from bioinspired.simulation import EarthSimulatorAdjustable
from bioinspired.spacecraft import Lander2
from bioinspired.spacecraft import SimpleCraft
from bioinspired.controllers import ConstantController


def quaternion_angular_error(q1, q2):
    """Compute the angular error between two quaternions in radians."""
    # Normalize quaternions
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)

    # Compute dot product - this is the cosine of half the angle between orientations
    dot_product = np.abs(np.dot(q1_norm, q2_norm))

    # Clamp to avoid numerical issues with arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Angular error in radians (multiply by 2 because quaternion represents half-angle)
    angular_error = 2.0 * np.arccos(np.abs(dot_product))

    return angular_error


def create_initial_state():
    """Create the initial state for the spacecraft."""
    initial_state = np.array([6378e3, 0, 0, 0, 8e3, 0])  # Initial position and velocity

    # Set initial rotation matrix (identity matrix)
    initial_rotation_matrix = np.eye(3)
    initial_rotational_velocity = np.array([0.0, 0, 0.1])  # Initial angular velocity

    # Set initial orientation by converting rotation matrix to quaternion
    initial_state_rotation = rotation_matrix_to_quaternion_entries(
        initial_rotation_matrix
    )

    # Complete initial state
    initial_state = np.concatenate(
        (initial_state, initial_state_rotation, initial_rotational_velocity)
    )

    return initial_state


def run_simulation_with_timestep(
    timestep,
    simulation_time=50.0,
    coefficient_set=integrator.CoefficientSets.rk_4,
    integrator_type="runge_kutta",
):
    """Run a simulation with a specific timestep, integrator and integrator type and return the final state."""
    if integrator_type == "runge_kutta":
        integrator_name = (
            coefficient_set.name
            if hasattr(coefficient_set, "name")
            else str(coefficient_set)
        )
        print(
            f"  Running simulation with timestep: {timestep:.6f}s, integrator: {integrator_type} ({integrator_name})"
        )

        # Create simulation with specific timestep, integrator, and integrator type
        simulator = EarthSimulatorAdjustable(
            stepsize=timestep,
            coefficient_set=coefficient_set,
            integrator_type=integrator_type,
        )
    else:
        # For ABM and Bulirsch-Stoer, coefficient sets are not used
        print(
            f"  Running simulation with timestep: {timestep:.6f}s, integrator: {integrator_type}"
        )

        # Create simulation with specific timestep and integrator type (coefficient_set ignored)
        simulator = EarthSimulatorAdjustable(
            stepsize=timestep, integrator_type=integrator_type
        )
        # For ABM and Bulirsch-Stoer, coefficient sets are not used
        print(
            f"  Running simulation with timestep: {timestep:.6f}s, integrator: {integrator_type}"
        )

        # Create simulation with specific timestep and integrator type (coefficient_set ignored)
        simulator = EarthSimulatorAdjustable(
            stepsize=timestep, integrator_type=integrator_type
        )

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
        initial_state=initial_state, simulation=simulator, controller=controller
    )

    # Run simulation
    start_time = time.perf_counter()
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=simulation_time)
    runtime = time.perf_counter() - start_time

    # Extract final state
    state_history = dynamics_simulator.propagation_results.state_history
    final_time = max(state_history.keys())
    final_state = state_history[final_time]

    # Clean up
    del simulator, spacecraft, dynamics_simulator

    return final_state, runtime


def create_benchmark_solution(simulation_time=50.0):
    """Create a high-resolution benchmark solution using forward Euler with 1ms timestep."""
    benchmark_timestep = 0.01  # 1 millisecond

    print(
        f"Creating benchmark solution using Forward Euler with {benchmark_timestep * 1000:.1f}ms timestep..."
    )
    print(
        f"This will take {int(simulation_time / benchmark_timestep)} integration steps..."
    )

    # Create benchmark simulation
    simulator = EarthSimulatorAdjustable(
        stepsize=benchmark_timestep,
        coefficient_set=integrator.CoefficientSets.rk_4,
        integrator_type="runge_kutta",
    )

    # Create initial state and controller
    initial_state = create_initial_state()
    controller = ConstantController(
        simulator=simulator,
        lander_name="Lander 2",
        target_name="Lander 2",
    )

    # Create spacecraft
    spacecraft = Lander2(
        initial_state=initial_state, simulation=simulator, controller=controller
    )

    # Run benchmark simulation
    start_time = time.perf_counter()
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=simulation_time)
    benchmark_runtime = time.perf_counter() - start_time

    # Extract final state
    state_history = dynamics_simulator.propagation_results.state_history
    final_time = max(state_history.keys())
    benchmark_final_state = state_history[final_time]

    print(f"✅ Benchmark solution created in {benchmark_runtime:.2f}s")
    print(f"   Final position: {benchmark_final_state[:3]}")
    print(f"   Final velocity: {benchmark_final_state[3:6]}")

    # Clean up
    del simulator, spacecraft, dynamics_simulator

    return benchmark_final_state, benchmark_runtime


def create_analytical_solution(simulation_time=50.0):
    """Create an analytical Keplerian solution for comparison."""
    from tudatpy.interface import spice
    from tudatpy import numerical_simulation
    from tudatpy.numerical_simulation import environment_setup, propagation_setup
    from tudatpy.astro import element_conversion
    from tudatpy import constants
    from tudatpy.util import result2array
    from tudatpy.astro.time_conversion import DateTime

    print("Setting up analytical Keplerian orbit solution...")

    # Load spice kernels
    spice.load_standard_kernels()

    # Create default body settings for Earth
    bodies_to_create = ["Earth"]
    global_frame_origin = "Earth"
    global_frame_orientation = "ECLIPJ2000"
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create, global_frame_origin, global_frame_orientation
    )

    # Add satellite settings
    body_settings.add_empty_settings("TestSat")

    # Create system of bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Get Earth gravitational parameter
    earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter

    # Define initial Keplerian elements (matching our simulation initial state)
    initial_state_keplerian = element_conversion.cartesian_to_keplerian(
        gravitational_parameter=earth_gravitational_parameter,
        cartesian_elements=create_initial_state()[:6].reshape(
            [6, 1]
        ),  # Use same initial state as simulation
    )

    print("Initial Keplerian elements:")
    print(f"  Semi-major axis: {initial_state_keplerian[0] / 1000:.2f} km")
    print(f"  Eccentricity: {initial_state_keplerian[1]:.6f}")
    print(f"  Inclination: {np.degrees(initial_state_keplerian[2]):.2f} deg")
    print(f"  Initial true anomaly: {np.degrees(initial_state_keplerian[5]):.2f} deg")

    # Convert initial true anomaly to mean anomaly for proper propagation
    initial_true_anomaly = initial_state_keplerian[5]
    initial_mean_anomaly = element_conversion.true_to_mean_anomaly(
        eccentricity=initial_state_keplerian[1], true_anomaly=initial_true_anomaly
    )

    print(f"  Initial mean anomaly: {np.degrees(initial_mean_anomaly):.2f} deg")

    # Create analytical solution by propagating Keplerian elements
    analytical_states = {}

    # Sample times throughout the simulation
    time_points = np.linspace(0, simulation_time, int(simulation_time / 0.1) + 1)

    for t in time_points:
        # Calculate mean anomaly at time t (starting from initial mean anomaly)
        mean_motion = np.sqrt(
            earth_gravitational_parameter / initial_state_keplerian[0] ** 3
        )
        mean_anomaly_t = initial_mean_anomaly + mean_motion * t

        # Convert mean anomaly to true anomaly (proper conversion for any eccentricity)
        true_anomaly_t = element_conversion.mean_to_true_anomaly(
            eccentricity=initial_state_keplerian[1], mean_anomaly=mean_anomaly_t
        )

        # Convert back to Cartesian coordinates
        state_t = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=earth_gravitational_parameter,
            semi_major_axis=initial_state_keplerian[0],
            eccentricity=initial_state_keplerian[1],
            inclination=initial_state_keplerian[2],
            argument_of_periapsis=initial_state_keplerian[3],
            longitude_of_ascending_node=initial_state_keplerian[4],
            true_anomaly=true_anomaly_t,
        )

        analytical_states[t] = state_t

    # Get final analytical state
    final_analytical_state = analytical_states[simulation_time]

    print("✅ Analytical solution created")
    print(f"   Final position: {final_analytical_state[:3]}")
    print(f"   Final velocity: {final_analytical_state[3:6]}")

    return final_analytical_state, analytical_states


def compute_state_error_vs_analytical(test_state, analytical_state):
    """Compute the error between a test state and the analytical solution."""
    # Only compare first 6 components (position and velocity) for Keplerian comparison
    test_pos_vel = test_state[:6]
    analytical_pos_vel = analytical_state[:6]

    # Compute position error (first 3 components)
    position_error = np.linalg.norm(test_pos_vel[:3] - analytical_pos_vel[:3])

    # Compute velocity error (next 3 components)
    velocity_error = np.linalg.norm(test_pos_vel[3:6] - analytical_pos_vel[3:6])

    # For rotational components, set to zero since analytical solution doesn't include rotation
    quat_error = 0.0
    angular_vel_error = 0.0
    angular_rate_error = 0.0
    orientation_error = 0.0

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "quaternion_error": quat_error,
        "angular_velocity_error": angular_vel_error,
        "angular_rate_error": angular_rate_error,
        "orientation_error": orientation_error,
    }


def compute_state_error_vs_benchmark(test_state, benchmark_state):
    """Compute the error between a test state and the benchmark solution."""
    # Compute position error (first 3 components)
    position_error = np.linalg.norm(test_state[:3] - benchmark_state[:3])

    # Compute velocity error (next 3 components)
    velocity_error = np.linalg.norm(test_state[3:6] - benchmark_state[3:6])

    # Compute quaternion error (next 4 components)
    quat_error = np.linalg.norm(test_state[6:10] - benchmark_state[6:10])

    # Compute angular velocity error (last 3 components)
    angular_vel_error = np.linalg.norm(test_state[10:13] - benchmark_state[10:13])

    # Compute angular rate error (magnitude of angular velocity vectors)
    test_angular_rate = np.linalg.norm(test_state[10:13])
    benchmark_angular_rate = np.linalg.norm(benchmark_state[10:13])
    angular_rate_error = abs(test_angular_rate - benchmark_angular_rate)

    # Compute orientation error from quaternions
    orientation_error = quaternion_angular_error(
        test_state[6:10], benchmark_state[6:10]
    )

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "quaternion_error": quat_error,
        "angular_velocity_error": angular_vel_error,
        "angular_rate_error": angular_rate_error,
        "orientation_error": orientation_error,
    }


def compute_state_error(state1, state2):
    """Compute the error between two states."""
    # Compute position error (first 3 components)
    position_error = np.linalg.norm(state1[:3] - state2[:3])

    # Compute velocity error (next 3 components)
    velocity_error = np.linalg.norm(state1[3:6] - state2[3:6])

    # Compute quaternion error (next 4 components)
    quat_error = np.linalg.norm(state1[6:10] - state2[6:10])

    # Compute angular velocity error (last 3 components)
    angular_vel_error = np.linalg.norm(state1[10:13] - state2[10:13])

    # Compute angular rate error (magnitude of angular velocity vectors)
    state1_angular_rate = np.linalg.norm(state1[10:13])
    state2_angular_rate = np.linalg.norm(state2[10:13])
    angular_rate_error = abs(state1_angular_rate - state2_angular_rate)

    # Compute orientation error from quaternions
    orientation_error = quaternion_angular_error(state1[6:10], state2[6:10])

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "quaternion_error": quat_error,
        "angular_velocity_error": angular_vel_error,
        "angular_rate_error": angular_rate_error,
        "orientation_error": orientation_error,
    }


def analyze_timestep_convergence(
    base_timesteps=None,
    simulation_time=50.0,
    coefficient_set=integrator.CoefficientSets.rk_4,
    integrator_type="runge_kutta",
    error_method="richardson",
):
    """Analyze how simulation error changes with timestep size.

    Args:
        base_timesteps: Array of timesteps to test
        simulation_time: Duration of simulation
        coefficient_set: For Runge-Kutta methods
        integrator_type: Type of integrator to use
        error_method: "richardson" for half-timestep comparison, "benchmark" for high-res benchmark, "analytical" for Keplerian solution
    """
    if base_timesteps is None:
        # Default timestep range from 5s down to 0.01s
        base_timesteps = np.array([5, 1, 0.5, 0.1, 0.05, 0.02, 0.01])

    if integrator_type == "runge_kutta":
        integrator_name = (
            coefficient_set.name
            if hasattr(coefficient_set, "name")
            else str(coefficient_set)
        )
        print("=" * 80)
        print(
            f"TIMESTEP CONVERGENCE ANALYSIS - {integrator_type.upper()} ({integrator_name.upper()})"
        )
    else:
        print("=" * 80)
        print(f"TIMESTEP CONVERGENCE ANALYSIS - {integrator_type.upper()}")

    print("=" * 80)
    print(f"Simulation time: {simulation_time}s")
    print(f"Error estimation method: {error_method.upper()}")

    if error_method == "richardson":
        print("Error estimation: Richardson extrapolation (compare with half-timestep)")
    elif error_method == "benchmark":
        print("Error estimation: High-resolution benchmark comparison")
    else:
        print("Error estimation: Analytical Keplerian solution comparison")

    print(f"Analyzing {len(base_timesteps)} timestep sizes...")

    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None
    analytical_states = None

    if error_method == "benchmark":
        # Create benchmark solution first
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        # Create analytical solution first
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, analytical_states = create_analytical_solution(
            simulation_time
        )

    print("\n" + "=" * 60)
    print("TESTING INTEGRATOR CONFIGURATIONS")
    print("=" * 60)

    results = []

    for i, timestep in enumerate(base_timesteps):
        print(f"\n--- Analysis {i + 1}/{len(base_timesteps)} ---")
        print(f"Timestep: {timestep:.6f}s")

        # Run simulation with current timestep
        final_state, runtime = run_simulation_with_timestep(
            timestep, simulation_time, coefficient_set, integrator_type
        )

        if error_method == "richardson":
            # Richardson extrapolation: compare with half-timestep
            print(f"  Running half-timestep simulation: {timestep / 2:.6f}s")
            final_state_half, runtime_half = run_simulation_with_timestep(
                timestep / 2, simulation_time, coefficient_set, integrator_type
            )

            # Compute error between full and half timestep
            error_metrics = compute_state_error(final_state, final_state_half)
            reference_runtime = runtime_half
        elif error_method == "benchmark":
            # Benchmark comparison
            error_metrics = compute_state_error_vs_benchmark(
                final_state, benchmark_final_state
            )
            reference_runtime = benchmark_runtime
        else:  # analytical
            # Analytical comparison
            analytical_final_state, _ = create_analytical_solution(simulation_time)
            error_metrics = compute_state_error_vs_analytical(
                final_state, analytical_final_state
            )
            reference_runtime = 0.0  # Analytical solution is instantaneous

        # Store results
        if integrator_type == "runge_kutta":
            integrator_name = (
                coefficient_set.name
                if hasattr(coefficient_set, "name")
                else str(coefficient_set)
            )
            integrator_display = f"{integrator_type} ({integrator_name})"
        else:
            integrator_display = integrator_type

        result = {
            "timestep": timestep,
            "runtime": runtime,
            "final_state": final_state,
            "integrator": integrator_display,
            "integrator_type": integrator_type,
            "error_method": error_method,
            "reference_runtime": reference_runtime,
            **error_metrics,
        }

        if error_method == "richardson":
            result["half_timestep_state"] = final_state_half
            result["half_timestep_runtime"] = runtime_half
        elif error_method == "benchmark":
            result["benchmark_final_state"] = benchmark_final_state
            result["benchmark_runtime"] = benchmark_runtime
        else:  # analytical
            result["analytical_final_state"] = analytical_final_state

        results.append(result)

        # Print summary
        if error_method == "richardson":
            print(f"  Runtime: {runtime:.4f}s (half-timestep: {runtime_half:.4f}s)")
            print(f"  Half-timestep overhead: {runtime_half / runtime:.1f}x")
        elif error_method == "benchmark":
            print(f"  Runtime: {runtime:.4f}s (vs benchmark: {benchmark_runtime:.2f}s)")
            print(
                f"  Speedup: {benchmark_runtime / runtime:.1f}x faster than benchmark"
            )
        else:  # analytical
            print(f"  Runtime: {runtime:.4f}s (vs analytical: instantaneous)")
            print("  Analytical comparison (perfect reference)")

        print(f"  Position error:     {error_metrics['position_error']:.2e} m")
        print(f"  Velocity error:     {error_metrics['velocity_error']:.2e} m/s")
        print(f"  Quaternion error:   {error_metrics['quaternion_error']:.2e}")
        print(
            f"  Angular vel error:  {error_metrics['angular_velocity_error']:.2e} rad/s"
        )
        print(f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s")
        print(f"  Orientation error:  {error_metrics['orientation_error']:.2e} rad")

    return results


def analyze_integrator_types_comparison(
    timesteps=None, simulation_time=50.0, error_method="richardson"
):
    """Compare different integrator types (RK vs ABM vs Bulirsch-Stoer) with selectable error estimation method."""
    if timesteps is None:
        timesteps = np.array([1.0, 0.5, 0.1, 0.05, 0.01])

    # Define best integrator combinations to test
    integrators = {
        "Runge-Kutta RK4": ("runge_kutta", integrator.CoefficientSets.rk_4),
        "Adams-Bashforth-Moulton": (
            "adams_bashforth_moulton",
            None,
        ),  # ABM doesn't use coefficient sets
        "Bulirsch-Stoer": ("bulirsch_stoer", None),  # BS doesn't use coefficient sets
    }

    print("=" * 80)
    print("INTEGRATOR TYPES COMPARISON ANALYSIS")
    print("=" * 80)
    print(
        f"Testing {len(integrators)} integrator types at {len(timesteps)} timestep sizes"
    )
    print(f"Integrator types: {', '.join(integrators.keys())}")
    print(f"Error estimation method: {error_method.upper()}")

    # Create reference solution based on error method
    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None
    analytical_states = None

    if error_method == "benchmark":
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, analytical_states = create_analytical_solution(
            simulation_time
        )

    print("\n" + "=" * 60)
    print("TESTING INTEGRATOR TYPES")
    print("=" * 60)

    all_results = {}

    for int_name, (int_type, coeff_set) in integrators.items():
        print(f"\n{'=' * 25} Testing {int_name} {'=' * 25}")

        # Test this integrator type at all timesteps
        integrator_results = []

        for timestep in timesteps:
            print(f"\n--- {int_name}: Timestep {timestep:.6f}s ---")

            try:
                # Run with current timestep (coefficient set only used for RK)
                final_state, runtime = run_simulation_with_timestep(
                    timestep,
                    simulation_time,
                    coeff_set or integrator.CoefficientSets.rk_4,
                    int_type,
                )

                # Compute errors based on error method
                if error_method == "richardson":
                    # Richardson extrapolation: compare with half-timestep
                    print(f"  Running half-timestep simulation: {timestep / 2:.6f}s")
                    final_state_half, runtime_half = run_simulation_with_timestep(
                        timestep / 2,
                        simulation_time,
                        coeff_set or integrator.CoefficientSets.rk_4,
                        int_type,
                    )

                    error_metrics = compute_state_error(final_state, final_state_half)
                    reference_runtime = runtime_half
                elif error_method == "benchmark":
                    error_metrics = compute_state_error_vs_benchmark(
                        final_state, benchmark_final_state
                    )
                    reference_runtime = benchmark_runtime
                else:  # analytical
                    error_metrics = compute_state_error_vs_analytical(
                        final_state, analytical_final_state
                    )
                    reference_runtime = 0.0

                result = {
                    "integrator": int_name,
                    "integrator_type": int_type,
                    "timestep": timestep,
                    "runtime": runtime,
                    "final_state": final_state,
                    "error_method": error_method,
                    "reference_runtime": reference_runtime,
                    **error_metrics,
                }

                # Add method-specific data
                if error_method == "richardson":
                    result["half_timestep_state"] = final_state_half
                    result["half_timestep_runtime"] = runtime_half
                elif error_method == "benchmark":
                    result["benchmark_final_state"] = benchmark_final_state
                    result["benchmark_runtime"] = benchmark_runtime
                else:  # analytical
                    result["analytical_final_state"] = analytical_final_state

                integrator_results.append(result)

                # Print runtime information based on method
                if error_method == "richardson":
                    print(
                        f"  Runtime: {runtime:.4f}s (half-timestep: {runtime_half:.4f}s)"
                    )
                    print(f"  Half-timestep overhead: {runtime_half / runtime:.1f}x")
                elif error_method == "benchmark":
                    print(
                        f"  Runtime: {runtime:.4f}s (vs benchmark: {benchmark_runtime:.2f}s)"
                    )
                    print(
                        f"  Speedup: {benchmark_runtime / runtime:.1f}x faster than benchmark"
                    )
                else:  # analytical
                    print(f"  Runtime: {runtime:.4f}s (vs analytical: instantaneous)")
                    print("  Analytical comparison (perfect reference)")

                print(f"  Position error: {error_metrics['position_error']:.2e} m")
                print(f"  Velocity error: {error_metrics['velocity_error']:.2e} m/s")
                print(
                    f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s"
                )
                print(
                    f"  Orientation error: {error_metrics['orientation_error']:.2e} rad"
                )

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                # Print more detailed error info for debugging
                import traceback

                print(f"  Full error: {traceback.format_exc()}")
                continue

        all_results[int_name] = integrator_results

    return all_results


def analyze_integrator_comparison(
    timesteps=None, simulation_time=50.0, error_method="richardson"
):
    """Compare different Runge-Kutta coefficient sets with selectable error estimation method."""
    if timesteps is None:
        timesteps = np.array([1.0, 0.5, 0.1, 0.05, 0.01])

    # Define Runge-Kutta integrators to test - only RK methods use coefficient sets
    integrators = {
        # 'RK - Euler Forward': ('runge_kutta', integrator.CoefficientSets.euler_forward),
        # 'RK - RK3': ('runge_kutta', integrator.CoefficientSets.rk_3),
        # 'RK - Ralston 3': ('runge_kutta', integrator.CoefficientSets.ralston_3),
        # 'RK - SSPRK3': ('runge_kutta', integrator.CoefficientSets.SSPRK3),
        # 'RK - RK4': ('runge_kutta', integrator.CoefficientSets.rk_4),
        # 'RK - Three-Eight RK4': ('runge_kutta', integrator.CoefficientSets.three_eight_rule_rk_4),
        # 'RK - RKF12': ('runge_kutta', integrator.CoefficientSets.rkf_12),
        "RK - RKF56": ("runge_kutta", integrator.CoefficientSets.rkf_56),
        "RK - RKF78": ("runge_kutta", integrator.CoefficientSets.rkf_78),
        "RK - RKF89": ("runge_kutta", integrator.CoefficientSets.rkf_89),
    }

    print("=" * 80)
    print("RUNGE-KUTTA COEFFICIENT SETS COMPARISON ANALYSIS")
    print("=" * 80)
    print(
        f"Testing {len(integrators)} Runge-Kutta coefficient sets at {len(timesteps)} timestep sizes"
    )
    print(
        f"Coefficient sets: {', '.join([name.split(' - ')[1] for name in integrators.keys()])}"
    )
    print(f"Error estimation method: {error_method.upper()}")

    # Create reference solution based on error method
    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None
    analytical_states = None

    if error_method == "benchmark":
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, analytical_states = create_analytical_solution(
            simulation_time
        )

    print("\n" + "=" * 60)
    print("TESTING RUNGE-KUTTA METHODS")
    print("=" * 60)

    all_results = {}

    for int_name, (int_type, coeff_set) in integrators.items():
        print(f"\n{'=' * 20} Testing {int_name} {'=' * 20}")

        # Test this integrator at all timesteps
        integrator_results = []

        for timestep in timesteps:
            print(f"\n--- {int_name}: Timestep {timestep:.6f}s ---")

            try:
                # Run with current timestep
                final_state, runtime = run_simulation_with_timestep(
                    timestep, simulation_time, coeff_set, int_type
                )

                # Compute errors based on error method
                if error_method == "richardson":
                    # Richardson extrapolation: compare with half-timestep
                    print(f"  Running half-timestep simulation: {timestep / 2:.6f}s")
                    final_state_half, runtime_half = run_simulation_with_timestep(
                        timestep / 2, simulation_time, coeff_set, int_type
                    )

                    error_metrics = compute_state_error(final_state, final_state_half)
                    reference_runtime = runtime_half
                elif error_method == "benchmark":
                    error_metrics = compute_state_error_vs_benchmark(
                        final_state, benchmark_final_state
                    )
                    reference_runtime = benchmark_runtime
                else:  # analytical
                    error_metrics = compute_state_error_vs_analytical(
                        final_state, analytical_final_state
                    )
                    reference_runtime = 0.0

                result = {
                    "integrator": int_name,
                    "integrator_type": int_type,
                    "timestep": timestep,
                    "runtime": runtime,
                    "final_state": final_state,
                    "error_method": error_method,
                    "reference_runtime": reference_runtime,
                    **error_metrics,
                }

                # Add method-specific data
                if error_method == "richardson":
                    result["half_timestep_state"] = final_state_half
                    result["half_timestep_runtime"] = runtime_half
                elif error_method == "benchmark":
                    result["benchmark_final_state"] = benchmark_final_state
                    result["benchmark_runtime"] = benchmark_runtime
                else:  # analytical
                    result["analytical_final_state"] = analytical_final_state

                integrator_results.append(result)

                # Print runtime information based on method
                if error_method == "richardson":
                    print(
                        f"  Runtime: {runtime:.4f}s (half-timestep: {runtime_half:.4f}s)"
                    )
                    print(f"  Half-timestep overhead: {runtime_half / runtime:.1f}x")
                elif error_method == "benchmark":
                    print(
                        f"  Runtime: {runtime:.4f}s (vs benchmark: {benchmark_runtime:.2f}s)"
                    )
                    print(
                        f"  Speedup: {benchmark_runtime / runtime:.1f}x faster than benchmark"
                    )
                else:  # analytical
                    print(f"  Runtime: {runtime:.4f}s (vs analytical: instantaneous)")
                    print("  Analytical comparison (perfect reference)")

                print(f"  Position error: {error_metrics['position_error']:.2e} m")
                print(f"  Velocity error: {error_metrics['velocity_error']:.2e} m/s")
                print(
                    f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s"
                )
                print(
                    f"  Orientation error: {error_metrics['orientation_error']:.2e} rad"
                )

            except Exception as e:
                print(f"  ❌ Failed: {str(e)}")
                continue

        all_results[int_name] = integrator_results

    return all_results


def analyze_bulirsch_stoer_parameters(
    timesteps=None, simulation_time=50.0, error_method="benchmark"
):
    """Perform parameter sweep on Bulirsch-Stoer integrator settings.

    Args:
        timesteps: Array of timesteps to test
        simulation_time: Duration of simulation
        error_method: "richardson", "benchmark", or "analytical"
    """
    if timesteps is None:
        # Focus on moderate timesteps where BS typically excels
        timesteps = np.array([1.0, 0.5, 0.1, 0.05, 0.01])

    # Define parameter combinations to test
    extrapolation_sequences = [
        integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
        integrator.ExtrapolationMethodStepSequences.deufelhard_sequence,
    ]

    max_steps_options = [3, 4, 5, 6]

    print("=" * 80)
    print("BULIRSCH-STOER PARAMETER SWEEP ANALYSIS")
    print("=" * 80)
    print(
        f"Testing {len(extrapolation_sequences)} extrapolation sequences x {len(max_steps_options)} max steps combinations"
    )
    print(
        "Sequences: bulirsch_stoer_sequence (2,4,6,8,12,16...), deufelhard_sequence (2,4,6,8,10,12,14...)"
    )
    print(f"Max steps: {max_steps_options}")
    print(f"Error estimation method: {error_method.upper()}")

    # Create reference solution based on error method
    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None
    analytical_states = None

    if error_method == "benchmark":
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, analytical_states = create_analytical_solution(
            simulation_time
        )

    print("\n" + "=" * 60)
    print("TESTING BULIRSCH-STOER PARAMETER COMBINATIONS")
    print("=" * 60)

    all_results = {}

    for seq in extrapolation_sequences:
        seq_name = (
            "bulirsch_stoer"
            if seq
            == integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
            else "deufelhard"
        )

        for max_steps in max_steps_options:
            config_name = f"BS-{seq_name}-{max_steps}steps"
            print(f"\n{'=' * 25} Testing {config_name} {'=' * 25}")

            integrator_results = []

            for timestep in timesteps:
                print(f"\n--- {config_name}: Timestep {timestep:.6f}s ---")

                try:
                    # Create simulator with specific BS parameters
                    simulator = EarthSimulatorAdjustable(
                        stepsize=timestep, integrator_type="bulirsch_stoer"
                    )

                    # Override the integrator creation to use our specific parameters
                    simulator._integrator = integrator.bulirsch_stoer_fixed_step(
                        timestep,
                        extrapolation_sequence=seq,
                        maximum_number_of_steps=max_steps,
                    )

                    # Create initial state and controller
                    initial_state = create_initial_state()
                    controller = ConstantController(
                        simulator=simulator,
                        lander_name="Lander 2",
                        target_name="Lander 2",
                    )

                    # Create spacecraft
                    spacecraft = Lander2(
                        initial_state=initial_state,
                        simulation=simulator,
                        controller=controller,
                    )

                    # Run simulation
                    start_time = time.perf_counter()
                    dynamics_simulator = simulator.run(
                        start_epoch=0.0, simulation_time=simulation_time
                    )
                    runtime = time.perf_counter() - start_time

                    # Extract final state
                    state_history = dynamics_simulator.propagation_results.state_history
                    final_time = max(state_history.keys())
                    final_state = state_history[final_time]

                    # Compute errors based on error method
                    if error_method == "richardson":
                        # Run with half timestep for Richardson comparison
                        half_timestep = timestep / 2
                        simulator_half = EarthSimulatorAdjustable(
                            stepsize=half_timestep, integrator_type="bulirsch_stoer"
                        )
                        simulator_half._integrator = (
                            integrator.bulirsch_stoer_fixed_step(
                                half_timestep,
                                extrapolation_sequence=seq,
                                maximum_number_of_steps=max_steps,
                            )
                        )

                        spacecraft_half = Lander2(
                            initial_state=initial_state,
                            simulation=simulator_half,
                            controller=ConstantController(
                                simulator=simulator_half,
                                lander_name="Lander 2",
                                target_name="Lander 2",
                            ),
                        )

                        start_time_half = time.perf_counter()
                        dynamics_simulator_half = simulator_half.run(
                            start_epoch=0.0, simulation_time=simulation_time
                        )
                        runtime_half = time.perf_counter() - start_time_half

                        state_history_half = (
                            dynamics_simulator_half.propagation_results.state_history
                        )
                        final_time_half = max(state_history_half.keys())
                        final_state_half = state_history_half[final_time_half]

                        error_metrics = compute_state_error(
                            final_state, final_state_half
                        )
                        reference_runtime = runtime_half

                        # Clean up
                        del simulator_half, spacecraft_half, dynamics_simulator_half

                    elif error_method == "benchmark":
                        error_metrics = compute_state_error_vs_benchmark(
                            final_state, benchmark_final_state
                        )
                        reference_runtime = benchmark_runtime

                    else:  # analytical
                        error_metrics = compute_state_error_vs_analytical(
                            final_state, analytical_final_state
                        )
                        reference_runtime = 0.0

                    result = {
                        "integrator": config_name,
                        "integrator_type": "bulirsch_stoer",
                        "extrapolation_sequence": seq_name,
                        "max_steps": max_steps,
                        "timestep": timestep,
                        "runtime": runtime,
                        "final_state": final_state,
                        "error_method": error_method,
                        "reference_runtime": reference_runtime,
                        **error_metrics,
                    }

                    # Add method-specific data
                    if error_method == "richardson":
                        result["half_timestep_state"] = final_state_half
                        result["half_timestep_runtime"] = runtime_half
                    elif error_method == "benchmark":
                        result["benchmark_final_state"] = benchmark_final_state
                        result["benchmark_runtime"] = benchmark_runtime
                    else:  # analytical
                        result["analytical_final_state"] = analytical_final_state

                    integrator_results.append(result)

                    # Print runtime information based on method
                    if error_method == "richardson":
                        print(
                            f"  Runtime: {runtime:.4f}s (half-timestep: {runtime_half:.4f}s)"
                        )
                        print(
                            f"  Half-timestep overhead: {runtime_half / runtime:.1f}x"
                        )
                    elif error_method == "benchmark":
                        print(
                            f"  Runtime: {runtime:.4f}s (vs benchmark: {benchmark_runtime:.2f}s)"
                        )
                        print(
                            f"  Speedup: {benchmark_runtime / runtime:.1f}x faster than benchmark"
                        )
                    else:  # analytical
                        print(
                            f"  Runtime: {runtime:.4f}s (vs analytical: instantaneous)"
                        )
                        print("  Analytical comparison (perfect reference)")

                    print(f"  Position error: {error_metrics['position_error']:.2e} m")
                    print(
                        f"  Velocity error: {error_metrics['velocity_error']:.2e} m/s"
                    )
                    print(
                        f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s"
                    )
                    print(
                        f"  Orientation error: {error_metrics['orientation_error']:.2e} rad"
                    )

                    # Clean up
                    del simulator, spacecraft, dynamics_simulator

                except Exception as e:
                    print(f"  ❌ Failed: {str(e)}")
                    continue

            all_results[config_name] = integrator_results

    return all_results


def analyze_bulirsch_stoer_variable_step_parameters(
    initial_timesteps=None, simulation_time=50.0, error_method="benchmark"
):
    """Perform parameter sweep on Bulirsch-Stoer variable stepsize integrator settings.

    Args:
        initial_timesteps: Array of initial timesteps to test
        simulation_time: Duration of simulation
        error_method: "richardson", "benchmark", or "analytical"
    """
    if initial_timesteps is None:
        # Variable step can start with larger initial timesteps since it adapts
        initial_timesteps = np.array([0.1])

    # Define parameter combinations to test
    extrapolation_sequences = [
        # integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
        integrator.ExtrapolationMethodStepSequences.deufelhard_sequence,
    ]

    max_steps_options = [3, 4, 5, 6]  # More steps for variable stepsize

    # Step size control parameters to test
    tolerance_options = [
        # 1e-3,
        1e-4,
        1e-5,
        1e-6,
        1e-7,
    ]  # Relative tolerance levels
    min_timestep_options = [1e-3]  # Minimum allowed timestep
    max_timestep_options = [10000]  # Maximum allowed timestep

    # Safety factors for step size control
    safety_factor_options = [0.8]
    max_increase_factor_options = [3]
    max_decrease_factor_options = [0.25]

    print("=" * 80)
    print("BULIRSCH-STOER VARIABLE STEPSIZE PARAMETER SWEEP ANALYSIS")
    print("=" * 80)
    print(f"Testing {len(extrapolation_sequences)} extrapolation sequences")
    print(f"Max steps options: {max_steps_options}")
    print(f"Tolerance levels: {tolerance_options}")
    print(f"Safety factors: {safety_factor_options}")
    print(f"Error estimation method: {error_method.upper()}")

    # Create reference solution based on error method
    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None

    if error_method == "benchmark":
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, _ = create_analytical_solution(simulation_time)

    print("\n" + "=" * 60)
    print("TESTING BULIRSCH-STOER VARIABLE STEPSIZE COMBINATIONS")
    print("=" * 60)

    all_results = {}

    for seq in extrapolation_sequences:
        seq_name = (
            "bulirsch_stoer"
            if seq
            == integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence
            else "deufelhard"
        )

        for max_steps in max_steps_options:
            for tolerance in tolerance_options:
                for safety_factor in safety_factor_options:
                    for max_increase in max_increase_factor_options:
                        config_name = f"BS-VAR-{seq_name}-{max_steps}steps-tol{tolerance:.0e}-sf{safety_factor:.1f}-inc{max_increase:.1f}"
                        print(f"\n{'=' * 30} Testing {config_name} {'=' * 30}")

                        integrator_results = []

                        for initial_timestep in initial_timesteps:
                            print(
                                f"\n--- {config_name}: Initial timestep {initial_timestep:.6f}s ---"
                            )

                            try:
                                block_indices = [
                                    (0, 0, 3, 1),
                                    (3, 0, 3, 1),
                                    (6, 0, 4, 1),
                                    (10, 0, 3, 1),
                                ]
                                step_size_control = integrator.step_size_control_blockwise_scalar_tolerance(
                                    block_indices, tolerance, tolerance
                                )

                                # Create step size validation settings
                                step_size_validation = integrator.step_size_validation(
                                    minimum_step=min_timestep_options[0],
                                    maximum_step=max_timestep_options[0],
                                )

                                # Create simulator with variable step BS integrator
                                simulator = EarthSimulatorAdjustable(
                                    stepsize=initial_timestep,  # This becomes the initial timestep
                                    integrator_type="bulirsch_stoer",
                                )

                                # Override with variable step integrator
                                simulator._integrator = integrator.bulirsch_stoer_variable_step(
                                    initial_time_step=initial_timestep,
                                    extrapolation_sequence=seq,
                                    # coefficient_set=integrator.CoefficientSets.rkf_78,
                                    maximum_number_of_steps=max_steps,
                                    step_size_control_settings=step_size_control,
                                    step_size_validation_settings=step_size_validation,
                                    assess_termination_on_minor_steps=False,
                                )

                                # Create initial state and controller
                                initial_state = create_initial_state()
                                controller = ConstantController(
                                    simulator=simulator,
                                    lander_name="Lander 2",
                                    target_name="Lander 2",
                                )

                                # Create spacecraft
                                spacecraft = Lander2(
                                    initial_state=initial_state,
                                    simulation=simulator,
                                    controller=controller,
                                )
                                # Run simulation
                                start_time = time.perf_counter()
                                dynamics_simulator = simulator.run(
                                    start_epoch=0.0, simulation_time=simulation_time
                                )
                                runtime = time.perf_counter() - start_time

                                # Extract final state
                                state_history = (
                                    dynamics_simulator.propagation_results.state_history
                                )
                                if (
                                    dynamics_simulator.integration_completed_successfully
                                    == False
                                ):
                                    raise RuntimeError(
                                        "Simulation did not complete successfully. Check the logs for details."
                                    )
                                final_time = max(state_history.keys())
                                final_state = state_history[final_time]

                                # Get integration statistics (for potential future use)
                                # cumulative_computation_time = dynamics_simulator.propagation_results.cumulative_computation_time
                                # dependent_variable_history = dynamics_simulator.propagation_results.dependent_variable_history

                                # Count number of function evaluations and steps taken
                                total_steps = len(state_history)

                                # Compute errors based on error method
                                if error_method == "richardson":
                                    # For variable step, Richardson is tricky - we'll use a much tighter tolerance
                                    print("  Running high-precision comparison...")

                                    # Create very high precision version
                                    block_indices_hp = [
                                        (0, 0, 3, 1),
                                        (3, 0, 3, 1),
                                        (6, 0, 4, 1),
                                        (10, 0, 3, 1),
                                    ]
                                    step_size_control_hp = integrator.step_size_control_blockwise_scalar_tolerance(
                                        block_indices_hp, tolerance/100, tolerance/100
                                    )

                                    # Create step size validation settings
                                    step_size_validatio_hp = (
                                        integrator.step_size_validation(
                                            minimum_step=min_timestep_options[0]/100,
                                            maximum_step=max_timestep_options[0],
                                        )
                                    )
                                    simulator_hp = EarthSimulatorAdjustable(
                                        stepsize=initial_timestep,
                                        integrator_type="bulirsch_stoer",
                                    )

                                    simulator_hp._integrator = integrator.bulirsch_stoer_variable_step(
                                        initial_time_step=initial_timestep / 10,
                                        extrapolation_sequence=seq,
                                        # coefficient_set=integrator.CoefficientSets.rkf_78,
                                        maximum_number_of_steps=max_steps,
                                        step_size_control_settings=step_size_control_hp,
                                        step_size_validation_settings=step_size_validatio_hp,
                                        assess_termination_on_minor_steps=False,
                                    )

                                    controller_hp = ConstantController(
                                        simulator=simulator_hp,
                                        lander_name="Lander 2",
                                        target_name="Lander 2",
                                    )

                                    spacecraft_hp = Lander2(
                                        initial_state=initial_state,
                                        simulation=simulator_hp,
                                        controller=controller_hp,
                                    )

                                    start_time_hp = time.perf_counter()
                                    dynamics_simulator_hp = simulator_hp.run(
                                        start_epoch=0.0, simulation_time=simulation_time
                                    )
                                    runtime_hp = time.perf_counter() - start_time_hp

                                    state_history_hp = dynamics_simulator_hp.propagation_results.state_history
                                    final_time_hp = max(state_history_hp.keys())
                                    final_state_hp = state_history_hp[final_time_hp]

                                    error_metrics = compute_state_error(
                                        final_state, final_state_hp
                                    )
                                    reference_runtime = runtime_hp

                                    # Clean up
                                    del (
                                        simulator_hp,
                                        spacecraft_hp,
                                        dynamics_simulator_hp,
                                    )

                                elif error_method == "benchmark":
                                    error_metrics = compute_state_error_vs_benchmark(
                                        final_state, benchmark_final_state
                                    )
                                    reference_runtime = benchmark_runtime

                                else:  # analytical
                                    error_metrics = compute_state_error_vs_analytical(
                                        final_state, analytical_final_state
                                    )
                                    reference_runtime = 0.0

                                result = {
                                    "integrator": config_name,
                                    "integrator_type": "bulirsch_stoer_variable",
                                    "extrapolation_sequence": seq_name,
                                    "max_steps": max_steps,
                                    "tolerance": tolerance,
                                    "safety_factor": safety_factor,
                                    "max_increase_factor": max_increase,
                                    "initial_timestep": initial_timestep,
                                    "runtime": runtime,
                                    "total_steps": total_steps,
                                    "final_state": final_state,
                                    "error_method": error_method,
                                    "reference_runtime": reference_runtime,
                                    "function_evaluations": dynamics_simulator.propagation_results.total_number_of_function_evaluations,
                                    **error_metrics,
                                }
                                print(
                                    f"Number of function evaluations: {result['function_evaluations']}"
                                )
                                print(f"Total steps taken: {total_steps}")
                                print(f"Tolerance: {tolerance:.0e}")
                                # Add method-specific data
                                if error_method == "richardson":
                                    result["high_precision_state"] = final_state_hp
                                    result["high_precision_runtime"] = runtime_hp
                                elif error_method == "benchmark":
                                    result["benchmark_final_state"] = (
                                        benchmark_final_state
                                    )
                                    result["benchmark_runtime"] = benchmark_runtime
                                else:  # analytical
                                    result["analytical_final_state"] = (
                                        analytical_final_state
                                    )

                                integrator_results.append(result)

                                # Print runtime information based on method
                                if error_method == "richardson":
                                    print(
                                        f"  Runtime: {runtime:.4f}s (high-precision: {runtime_hp:.4f}s)"
                                    )
                                    print(
                                        f"  High-precision overhead: {runtime_hp / runtime:.1f}x"
                                    )
                                elif error_method == "benchmark":
                                    print(
                                        f"  Runtime: {runtime:.4f}s (vs benchmark: {benchmark_runtime:.2f}s)"
                                    )
                                    print(
                                        f"  Speedup: {benchmark_runtime / runtime:.1f}x faster than benchmark"
                                    )
                                else:  # analytical
                                    print(
                                        f"  Runtime: {runtime:.4f}s (vs analytical: instantaneous)"
                                    )

                                print(f"  Total integration steps: {total_steps}")
                                print(
                                    f"  Average step size: {simulation_time / total_steps:.6f}s"
                                )
                                print(
                                    f"  Position error: {error_metrics['position_error']:.2e} m"
                                )
                                print(
                                    f"  Velocity error: {error_metrics['velocity_error']:.2e} m/s"
                                )
                                print(
                                    f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s"
                                )
                                print(
                                    f"  Orientation error: {error_metrics['orientation_error']:.2e} rad"
                                )

                                # Clean up
                                del simulator, spacecraft, dynamics_simulator

                            except Exception as e:
                                print(f"  ❌ Failed: {str(e)}")
                                continue

                        all_results[config_name] = integrator_results

    print("\n✅ Variable stepsize Bulirsch-Stoer parameter sweep completed")
    print(f"   Total configurations tested: {len(all_results)}")

    return all_results


def analyze_monte_carlo_sensitivity(
    timestep=0.1,
    integrator_type="runge_kutta",
    coefficient_set=integrator.CoefficientSets.rk_4,
    simulation_time=50.0,
    n_samples=100,
    error_method="benchmark",
):
    """Perform Monte Carlo analysis to test system sensitivity to initial conditions and control vectors.

    Args:
        timestep: Fixed timestep to use for all simulations
        integrator_type: Type of integrator to use
        coefficient_set: For Runge-Kutta methods
        simulation_time: Duration of simulation
        n_samples: Number of Monte Carlo samples
        error_method: "richardson", "benchmark", or "analytical"
    """

    print("=" * 80)
    print("MONTE CARLO SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Fixed timestep: {timestep:.6f}s")
    print(f"Integrator: {integrator_type}")
    if integrator_type == "runge_kutta":
        coeff_name = (
            coefficient_set.name
            if hasattr(coefficient_set, "name")
            else str(coefficient_set)
        )
        print(f"Coefficient set: {coeff_name}")
    print(f"Simulation time: {simulation_time}s")
    print(f"Monte Carlo samples: {n_samples}")
    print(f"Error estimation method: {error_method.upper()}")

    # Create reference solution based on error method
    benchmark_final_state = None
    benchmark_runtime = None
    analytical_final_state = None

    if error_method == "benchmark":
        print("\n" + "=" * 60)
        print("CREATING BENCHMARK SOLUTION")
        print("=" * 60)
        benchmark_final_state, benchmark_runtime = create_benchmark_solution(
            simulation_time
        )
    elif error_method == "analytical":
        print("\n" + "=" * 60)
        print("CREATING ANALYTICAL SOLUTION")
        print("=" * 60)
        analytical_final_state, _ = create_analytical_solution(simulation_time)

    print("\n" + "=" * 60)
    print("RUNNING MONTE CARLO SIMULATIONS")
    print("=" * 60)

    results = []

    for i in range(n_samples):
        print(f"\n--- Monte Carlo Sample {i + 1}/{n_samples} ---")

        try:
            # Generate random initial state variations
            # Base initial state
            base_initial_state = create_initial_state()

            # Add random variations to initial position (±10% of Earth radius ~ 600 km)
            position_variation = np.random.uniform(
                -6e5, 6e5, 3
            )  # ±600 km in each direction
            velocity_variation = np.random.uniform(
                -1e3, 1e3, 3
            )  # ±1 km/s in each direction

            # Add random variations to initial orientation (small random rotation)
            angle_variation = np.random.uniform(0, 0.1)  # Small angle in radians
            axis_variation = np.random.uniform(-1, 1, 3)
            axis_variation = axis_variation / np.linalg.norm(
                axis_variation
            )  # Normalize

            # Convert axis-angle to quaternion variation
            quat_variation = np.array(
                [
                    np.cos(angle_variation / 2),
                    axis_variation[0] * np.sin(angle_variation / 2),
                    axis_variation[1] * np.sin(angle_variation / 2),
                    axis_variation[2] * np.sin(angle_variation / 2),
                ]
            )

            # Add random angular velocity variation
            angular_vel_variation = np.random.uniform(-0.1, 0.1, 3)  # ±0.1 rad/s

            # Create varied initial state
            varied_initial_state = base_initial_state.copy()
            varied_initial_state[0:3] += position_variation
            varied_initial_state[3:6] += velocity_variation
            varied_initial_state[6:10] = (
                quat_variation  # Replace with varied quaternion
            )
            varied_initial_state[10:13] += angular_vel_variation

            # Generate random control vector (values between 0 and 1)
            control_vector = np.random.uniform(0, 1, 24)  # 24-element control vector

            # Create simulator
            if integrator_type == "runge_kutta":
                simulator = EarthSimulatorAdjustable(
                    stepsize=timestep,
                    coefficient_set=coefficient_set,
                    integrator_type=integrator_type,
                )
            else:
                simulator = EarthSimulatorAdjustable(
                    stepsize=timestep, integrator_type=integrator_type
                )

            # Create controller with random control vector
            controller = ConstantController(
                simulator=simulator,
                lander_name="Lander 2",
                target_name="Lander 2",
                control_vector=control_vector,
            )

            # Create spacecraft
            spacecraft = Lander2(
                initial_state=varied_initial_state,
                simulation=simulator,
                controller=controller,
            )

            # Run simulation
            start_time = time.perf_counter()
            dynamics_simulator = simulator.run(
                start_epoch=0.0, simulation_time=simulation_time
            )
            runtime = time.perf_counter() - start_time

            # Extract final state
            state_history = dynamics_simulator.propagation_results.state_history
            final_time = max(state_history.keys())
            final_state = state_history[final_time]

            # Compute errors based on error method
            if error_method == "richardson":
                # Run with half timestep for Richardson comparison
                half_timestep = timestep / 2
                if integrator_type == "runge_kutta":
                    simulator_half = EarthSimulatorAdjustable(
                        stepsize=half_timestep,
                        coefficient_set=coefficient_set,
                        integrator_type=integrator_type,
                    )
                else:
                    simulator_half = EarthSimulatorAdjustable(
                        stepsize=half_timestep, integrator_type=integrator_type
                    )

                controller_half = ConstantController(
                    simulator=simulator_half,
                    lander_name="Lander 2",
                    target_name="Lander 2",
                    control_vector=control_vector,
                )

                spacecraft_half = Lander2(
                    initial_state=varied_initial_state,
                    simulation=simulator_half,
                    controller=controller_half,
                )

                start_time_half = time.perf_counter()
                dynamics_simulator_half = simulator_half.run(
                    start_epoch=0.0, simulation_time=simulation_time
                )
                runtime_half = time.perf_counter() - start_time_half

                state_history_half = (
                    dynamics_simulator_half.propagation_results.state_history
                )
                final_time_half = max(state_history_half.keys())
                final_state_half = state_history_half[final_time_half]

                error_metrics = compute_state_error(final_state, final_state_half)
                reference_runtime = runtime_half

                # Clean up
                del simulator_half, spacecraft_half, dynamics_simulator_half

            elif error_method == "benchmark":
                error_metrics = compute_state_error_vs_benchmark(
                    final_state, benchmark_final_state
                )
                reference_runtime = benchmark_runtime

            else:  # analytical
                error_metrics = compute_state_error_vs_analytical(
                    final_state, analytical_final_state
                )
                reference_runtime = 0.0

            # Store result
            result = {
                "sample_id": i,
                "timestep": timestep,
                "integrator_type": integrator_type,
                "runtime": runtime,
                "final_state": final_state,
                "initial_state": varied_initial_state,
                "control_vector": control_vector,
                "error_method": error_method,
                "reference_runtime": reference_runtime,
                **error_metrics,
            }

            # Add method-specific data
            if error_method == "richardson":
                result["half_timestep_state"] = final_state_half
                result["half_timestep_runtime"] = runtime_half
            elif error_method == "benchmark":
                result["benchmark_final_state"] = benchmark_final_state
                result["benchmark_runtime"] = benchmark_runtime
            else:  # analytical
                result["analytical_final_state"] = analytical_final_state

            results.append(result)

            # Print basic info for this sample
            print(f"  Position error: {error_metrics['position_error']:.2e} m")
            print(f"  Velocity error: {error_metrics['velocity_error']:.2e} m/s")
            print(
                f"  Angular rate error: {error_metrics['angular_rate_error']:.2e} rad/s"
            )
            print(f"  Orientation error: {error_metrics['orientation_error']:.2e} rad")
            print(f"  Runtime: {runtime:.4f}s")

            # Clean up
            del simulator, spacecraft, dynamics_simulator

        except Exception as e:
            print(f"  ❌ Sample {i + 1} failed: {str(e)}")
            continue

    print(
        f"\n✅ Monte Carlo analysis completed: {len(results)}/{n_samples} successful samples"
    )

    return results


def plot_monte_carlo_results(results, save_plot=True):
    """Plot Monte Carlo sensitivity analysis results with all scatter plots."""

    if not results:
        print("No results to plot")
        return None

    # Extract data
    error_method = results[0]["error_method"]
    timestep = results[0]["timestep"]
    integrator_type = results[0]["integrator_type"]

    # Data arrays
    position_errors = [r["position_error"] for r in results]
    velocity_errors = [r["velocity_error"] for r in results]
    orientation_errors = [r["orientation_error"] for r in results]
    angular_rate_errors = [r["angular_rate_error"] for r in results]
    quaternion_errors = [r["quaternion_error"] for r in results]
    runtimes = [r["runtime"] for r in results]

    # Create figure with 2x3 layout - all scatter plots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 12))

    if error_method == "richardson":
        title_suffix = "(Richardson Extrapolation)"
    elif error_method == "benchmark":
        title_suffix = "(vs High-Resolution Benchmark)"
    else:  # analytical
        title_suffix = "(vs Analytical Keplerian Solution)"

    fig.suptitle(
        f"Monte Carlo Sensitivity Analysis - {integrator_type} @ {timestep:.6f}s {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    # Generate sample indices for x-axis
    sample_ids = [r["sample_id"] for r in results]

    # Plot 1: Position Errors
    ax1.scatter(sample_ids, position_errors, alpha=0.6, s=30)
    ax1.set_xlabel("Monte Carlo Sample ID")
    ax1.set_ylabel(f"Position Error vs {error_method.title()} (m)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Position Error Distribution")

    # Plot 2: Velocity Errors
    ax2.scatter(sample_ids, velocity_errors, alpha=0.6, s=30, color="orange")
    ax2.set_xlabel("Monte Carlo Sample ID")
    ax2.set_ylabel(f"Velocity Error vs {error_method.title()} (m/s)")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Velocity Error Distribution")

    # Plot 3: Orientation Errors
    ax3.scatter(sample_ids, orientation_errors, alpha=0.6, s=30, color="green")
    ax3.set_xlabel("Monte Carlo Sample ID")
    ax3.set_ylabel("Orientation Error (rad)")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Orientation Error Distribution")

    # Plot 4: Angular Rate Errors
    ax4.scatter(sample_ids, angular_rate_errors, alpha=0.6, s=30, color="red")
    ax4.set_xlabel("Monte Carlo Sample ID")
    ax4.set_ylabel("Angular Rate Error (rad/s)")
    ax4.set_yscale("log")
    ax4.grid(True, alpha=0.3)
    ax4.set_title("Angular Rate Error Distribution")

    # Plot 5: Runtime Distribution
    ax5.scatter(sample_ids, runtimes, alpha=0.6, s=30, color="purple")
    ax5.set_xlabel("Monte Carlo Sample ID")
    ax5.set_ylabel("Runtime (s)")
    ax5.grid(True, alpha=0.3)
    ax5.set_title("Runtime Distribution")

    # Plot 6: Error vs Runtime Trade-off
    ax6.scatter(runtimes, quaternion_errors, alpha=0.6, s=30, color="brown")
    ax6.set_xlabel("Runtime (s)")
    ax6.set_ylabel(f"Quaternion Error vs {error_method.title()}")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)
    ax6.set_title("Error vs Runtime Trade-off")

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monte_carlo_sensitivity_{integrator_type}_{timestep:.6f}s_{error_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

    plt.show()
    return fig


def print_monte_carlo_summary(results):
    """Print summary statistics of Monte Carlo sensitivity analysis."""

    if not results:
        print("No results to summarize")
        return

    error_method = results[0]["error_method"]
    timestep = results[0]["timestep"]
    integrator_type = results[0]["integrator_type"]

    print("\n" + "=" * 80)
    print("MONTE CARLO SENSITIVITY SUMMARY")
    print("=" * 80)
    print(f"Integrator: {integrator_type}")
    print(f"Fixed timestep: {timestep:.6f}s")
    print(f"Samples: {len(results)}")
    print(f"Error method: {error_method.upper()}")

    # Extract error data
    position_errors = np.array([r["position_error"] for r in results])
    velocity_errors = np.array([r["velocity_error"] for r in results])
    orientation_errors = np.array([r["orientation_error"] for r in results])
    angular_rate_errors = np.array([r["angular_rate_error"] for r in results])
    runtimes = np.array([r["runtime"] for r in results])

    print("\nPOSITION ERROR STATISTICS:")
    print(f"  Mean: {np.mean(position_errors):.2e} m")
    print(f"  Std:  {np.std(position_errors):.2e} m")
    print(f"  Min:  {np.min(position_errors):.2e} m")
    print(f"  Max:  {np.max(position_errors):.2e} m")
    print(
        f"  CV:   {np.std(position_errors) / np.mean(position_errors):.3f} (coefficient of variation)"
    )

    print("\nVELOCITY ERROR STATISTICS:")
    print(f"  Mean: {np.mean(velocity_errors):.2e} m/s")
    print(f"  Std:  {np.std(velocity_errors):.2e} m/s")
    print(f"  Min:  {np.min(velocity_errors):.2e} m/s")
    print(f"  Max:  {np.max(velocity_errors):.2e} m/s")
    print(f"  CV:   {np.std(velocity_errors) / np.mean(velocity_errors):.3f}")

    print("\nORIENTATION ERROR STATISTICS:")
    print(f"  Mean: {np.mean(orientation_errors):.2e} rad")
    print(f"  Std:  {np.std(orientation_errors):.2e} rad")
    print(f"  Min:  {np.min(orientation_errors):.2e} rad")
    print(f"  Max:  {np.max(orientation_errors):.2e} rad")
    print(f"  CV:   {np.std(orientation_errors) / np.mean(orientation_errors):.3f}")

    print("\nANGULAR RATE ERROR STATISTICS:")
    print(f"  Mean: {np.mean(angular_rate_errors):.2e} rad/s")
    print(f"  Std:  {np.std(angular_rate_errors):.2e} rad/s")
    print(f"  Min:  {np.min(angular_rate_errors):.2e} rad/s")
    print(f"  Max:  {np.max(angular_rate_errors):.2e} rad/s")
    print(f"  CV:   {np.std(angular_rate_errors) / np.mean(angular_rate_errors):.3f}")

    print("\nRUNTIME STATISTICS:")
    print(f"  Mean: {np.mean(runtimes):.4f} s")
    print(f"  Std:  {np.std(runtimes):.4f} s")
    print(f"  Min:  {np.min(runtimes):.4f} s")
    print(f"  Max:  {np.max(runtimes):.4f} s")
    print(f"  CV:   {np.std(runtimes) / np.mean(runtimes):.3f}")

    # Sensitivity assessment
    print("\nSENSITIVITY ASSESSMENT:")
    pos_cv = np.std(position_errors) / np.mean(position_errors)
    vel_cv = np.std(velocity_errors) / np.mean(velocity_errors)
    ori_cv = np.std(orientation_errors) / np.mean(orientation_errors)
    ang_cv = np.std(angular_rate_errors) / np.mean(angular_rate_errors)

    def sensitivity_level(cv):
        if cv < 0.1:
            return "LOW 🟢"
        elif cv < 0.5:
            return "MODERATE 🟡"
        else:
            return "HIGH 🔴"

    print(f"  Position sensitivity: {sensitivity_level(pos_cv)} (CV = {pos_cv:.3f})")
    print(f"  Velocity sensitivity: {sensitivity_level(vel_cv)} (CV = {vel_cv:.3f})")
    print(f"  Orientation sensitivity: {sensitivity_level(ori_cv)} (CV = {ori_cv:.3f})")
    print(
        f"  Angular rate sensitivity: {sensitivity_level(ang_cv)} (CV = {ang_cv:.3f})"
    )

    print("\nInterpretation:")
    print("  CV < 0.1: Low sensitivity - system is robust to parameter variations")
    print("  CV 0.1-0.5: Moderate sensitivity - some sensitivity present")
    print(
        "  CV > 0.5: High sensitivity - system highly sensitive to parameter variations"
    )


def print_bulirsch_stoer_parameter_summary(all_results):
    """Print a summary table of the Bulirsch-Stoer parameter sweep analysis."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    print("\n" + "=" * 140)
    print(f"BULIRSCH-STOER PARAMETER SWEEP SUMMARY (vs {error_method.upper()})")
    print("=" * 140)

    # For each timestep, compare parameter combinations
    if not all_results:
        print("No results to display")
        return

    # Get all timesteps that were tested
    all_timesteps = set()
    for results in all_results.values():
        for r in results:
            all_timesteps.add(r["timestep"])
    all_timesteps = sorted(all_timesteps, reverse=True)

    for timestep in all_timesteps:
        print(f"\n--- TIMESTEP: {timestep:.6f}s ---")

        if error_method == "richardson":
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Half-step RT':<12}"
        elif error_method == "benchmark":
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Speedup':<10}"
        else:  # analytical
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Efficiency':<12}"

        print(header)
        print("-" * 140)

        timestep_results = []
        for config_name, results in all_results.items():
            for r in results:
                if abs(r["timestep"] - timestep) < 1e-10:  # Match timestep
                    if error_method == "benchmark":
                        benchmark_runtime = r.get(
                            "reference_runtime", r.get("benchmark_runtime", 1.0)
                        )
                        metric = benchmark_runtime / r["runtime"]  # speedup
                    elif error_method == "richardson":
                        metric = r.get(
                            "half_timestep_runtime", r.get("reference_runtime", 0)
                        )  # half-step runtime
                    else:  # analytical
                        efficiency = (
                            (1.0 / r["position_error"]) / r["runtime"]
                            if r["position_error"] > 0
                            else 0
                        )
                        metric = efficiency

                    timestep_results.append(
                        {
                            "configuration": config_name,
                            "position_error": r["position_error"],
                            "velocity_error": r["velocity_error"],
                            "orientation_error": r["orientation_error"],
                            "runtime": r["runtime"],
                            "metric": metric,
                            "extrapolation_sequence": r["extrapolation_sequence"],
                            "max_steps": r["max_steps"],
                        }
                    )
                    break

        # Sort by position error (most accurate first)
        timestep_results.sort(key=lambda x: x["position_error"])

        for i, r in enumerate(timestep_results):
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "

            if error_method == "benchmark":
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<10.1f}x"
            elif error_method == "richardson":
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<12.4f}"
            else:  # analytical
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<12.2e}"

            print(row)

    # Overall parameter recommendations
    print("\n" + "=" * 140)
    print(f"BULIRSCH-STOER PARAMETER RECOMMENDATIONS (vs {error_method.upper()})")
    print("=" * 140)

    # Find best parameter combination for different criteria
    best_accuracy = {}
    best_speed = {}
    best_efficiency = {}

    for config_name, results in all_results.items():
        if not results:
            continue

        # Best accuracy (smallest position error)
        best_acc_result = min(results, key=lambda x: x["position_error"])
        best_accuracy[config_name] = best_acc_result

        # Best speed (fastest runtime at any timestep)
        best_speed_result = min(results, key=lambda x: x["runtime"])
        best_speed[config_name] = best_speed_result

        # Best efficiency (accuracy/runtime ratio)
        for r in results:
            r["efficiency"] = (
                (1.0 / r["position_error"]) / r["runtime"]
                if r["position_error"] > 0
                else 0
            )
        best_eff_result = max(results, key=lambda x: x["efficiency"])
        best_efficiency[config_name] = best_eff_result

    if best_accuracy:
        most_accurate = min(best_accuracy.items(), key=lambda x: x[1]["position_error"])
        print(
            f"🎯 Most Accurate: {most_accurate[0]} (pos error: {most_accurate[1]['position_error']:.2e} m at timestep {most_accurate[1]['timestep']:.6f}s)"
        )
        print(
            f"   Sequence: {most_accurate[1]['extrapolation_sequence']}, Max steps: {most_accurate[1]['max_steps']}"
        )

    if best_speed:
        fastest = min(best_speed.items(), key=lambda x: x[1]["runtime"])
        print(
            f"⚡ Fastest: {fastest[0]} (runtime: {fastest[1]['runtime']:.4f}s at timestep {fastest[1]['timestep']:.6f}s)"
        )
        print(
            f"   Sequence: {fastest[1]['extrapolation_sequence']}, Max steps: {fastest[1]['max_steps']}"
        )

    if best_efficiency:
        most_efficient = max(best_efficiency.items(), key=lambda x: x[1]["efficiency"])
        print(
            f"⚖️  Most Efficient: {most_efficient[0]} (efficiency: {most_efficient[1]['efficiency']:.2e} at timestep {most_efficient[1]['timestep']:.6f}s)"
        )
        print(
            f"   Sequence: {most_efficient[1]['extrapolation_sequence']}, Max steps: {most_efficient[1]['max_steps']}"
        )

    # Parameter-specific insights
    print("\nPARAMETER INSIGHTS:")
    print(
        "📊 Bulirsch-Stoer sequence (2,4,6,8,12,16...): Traditional sequence, well-tested"
    )
    print("📊 Deufelhard sequence (2,4,6,8,10,12,14...): More uniform step progression")
    print("📊 Higher max_steps: Better accuracy but increased computational cost")
    print("📊 Lower max_steps: Faster execution but may sacrifice accuracy")

    # Error method specific notes
    if error_method == "richardson":
        print(
            "\n📊 Note: Richardson extrapolation shows relative convergence between h and h/2 timesteps"
        )
    elif error_method == "benchmark":
        print(
            "\n📊 Note: Benchmark comparison shows absolute accuracy vs high-resolution Forward Euler solution"
        )
    else:  # analytical
        print(
            "\n📊 Note: Analytical comparison shows absolute accuracy vs perfect Keplerian orbital solution"
        )


def print_bulirsch_stoer_variable_step_summary(all_results):
    """Print a summary table of the Bulirsch-Stoer variable stepsize parameter sweep analysis."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    print("\n" + "=" * 150)
    print(
        f"BULIRSCH-STOER VARIABLE STEPSIZE PARAMETER SWEEP SUMMARY (vs {error_method.upper()})"
    )
    print("=" * 150)

    # For variable stepsize, we group by initial timestep and tolerance combinations
    if not all_results:
        print("No results to display")
        return

    # Get all initial timesteps and tolerances that were tested
    all_initial_timesteps = set()
    all_tolerances = set()
    for results in all_results.values():
        for r in results:
            # Variable stepsize results use "initial_timestep" instead of "timestep"
            initial_timestep_key = (
                "initial_timestep" if "initial_timestep" in r else "timestep"
            )
            all_initial_timesteps.add(r[initial_timestep_key])
            # Get tolerance if available
            if "tolerance" in r:
                all_tolerances.add(r["tolerance"])
            elif "relative_tolerance" in r:
                all_tolerances.add(r["relative_tolerance"])

    all_initial_timesteps = sorted(all_initial_timesteps, reverse=True)
    all_tolerances = sorted(all_tolerances)

    # Group results by initial timestep
    for initial_timestep in all_initial_timesteps:
        print(f"\n--- INITIAL TIMESTEP: {initial_timestep:.6f}s ---")

        if error_method == "richardson":
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Func Evals':<12} {'Avg Step (s)':<12}"
        elif error_method == "benchmark":
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Func Evals':<12} {'Speedup':<10}"
        else:  # analytical
            header = f"{'Configuration':<25} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Func Evals':<12} {'Efficiency':<12}"

        print(header)
        print("-" * 150)

        initial_timestep_results = []
        for config_name, results in all_results.items():
            for r in results:
                # Handle both "initial_timestep" and "timestep" keys
                timestep_key = (
                    "initial_timestep" if "initial_timestep" in r else "timestep"
                )
                if (
                    abs(r[timestep_key] - initial_timestep) < 1e-10
                ):  # Match initial timestep
                    # Calculate metrics
                    if error_method == "benchmark":
                        benchmark_runtime = r.get(
                            "reference_runtime", r.get("benchmark_runtime", 1.0)
                        )
                        metric = benchmark_runtime / r["runtime"]  # speedup
                    elif error_method == "richardson":
                        # For variable stepsize, show average stepsize instead of half-step runtime
                        avg_stepsize = r.get(
                            "average_stepsize",
                            r.get("final_time", 50.0) / r.get("num_steps", 1),
                        )
                        metric = avg_stepsize
                    else:  # analytical
                        efficiency = (
                            (1.0 / r["position_error"]) / r["runtime"]
                            if r["position_error"] > 0
                            else 0
                        )
                        metric = efficiency

                    # Get function evaluations if available
                    func_evals = r.get(
                        "function_evaluations", r.get("num_function_evaluations", "N/A")
                    )

                    initial_timestep_results.append(
                        {
                            "configuration": config_name,
                            "position_error": r["position_error"],
                            "velocity_error": r["velocity_error"],
                            "orientation_error": r["orientation_error"],
                            "runtime": r["runtime"],
                            "metric": metric,
                            "function_evaluations": func_evals,
                            "extrapolation_sequence": r.get(
                                "extrapolation_sequence", "Unknown"
                            ),
                            "max_steps": r.get("max_steps", "Unknown"),
                            "tolerance": r.get(
                                "tolerance", r.get("relative_tolerance", "Unknown")
                            ),
                            "average_stepsize": r.get(
                                "average_stepsize",
                                r.get("final_time", 50.0) / r.get("num_steps", 1),
                            ),
                        }
                    )
                    break

        # Sort by position error (most accurate first)
        initial_timestep_results.sort(key=lambda x: x["position_error"])

        for i, r in enumerate(initial_timestep_results):
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "

            func_eval_str = (
                f"{r['function_evaluations']}"
                if isinstance(r["function_evaluations"], (int, float))
                else str(r["function_evaluations"])[:10]
            )

            if error_method == "benchmark":
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {func_eval_str:<12} {r['metric']:<10.1f}x"
            elif error_method == "richardson":
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {func_eval_str:<12} {r['metric']:<12.6f}"
            else:  # analytical
                row = f"{marker} {r['configuration']:<23} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {func_eval_str:<12} {r['metric']:<12.2e}"

            print(row)
            # Add parameter details on next line
            tolerance_str = (
                f"{r['tolerance']}"
                if isinstance(r["tolerance"], (int, float))
                else str(r["tolerance"])
            )
            print(
                f"    Tolerance: {tolerance_str}, Avg Step: {r['average_stepsize']:.6f}s, Seq: {r['extrapolation_sequence']}, Max: {r['max_steps']}"
            )

    # Overall parameter recommendations for variable stepsize
    print("\n" + "=" * 150)
    print(
        f"BULIRSCH-STOER VARIABLE STEPSIZE PARAMETER RECOMMENDATIONS (vs {error_method.upper()})"
    )
    print("=" * 150)

    # Find best parameter combination for different criteria
    best_accuracy = {}
    best_speed = {}
    best_efficiency = {}
    best_func_evals = {}

    for config_name, results in all_results.items():
        if not results:
            continue

        # Best accuracy (smallest position error)
        best_acc_result = min(results, key=lambda x: x["position_error"])
        best_accuracy[config_name] = best_acc_result

        # Best speed (fastest runtime)
        best_speed_result = min(results, key=lambda x: x["runtime"])
        best_speed[config_name] = best_speed_result

        # Best efficiency (accuracy/runtime ratio)
        for r in results:
            r["efficiency"] = (
                (1.0 / r["position_error"]) / r["runtime"]
                if r["position_error"] > 0
                else 0
            )
        best_eff_result = max(results, key=lambda x: x["efficiency"])
        best_efficiency[config_name] = best_eff_result

        # Best function evaluation efficiency (fewest function calls for given accuracy)
        for r in results:
            func_evals = r.get(
                "function_evaluations", r.get("num_function_evaluations", float("inf"))
            )
            if isinstance(func_evals, (int, float)) and func_evals > 0:
                r["func_eval_efficiency"] = (1.0 / r["position_error"]) / func_evals
            else:
                r["func_eval_efficiency"] = 0
        best_func_result = max(
            [r for r in results if r.get("func_eval_efficiency", 0) > 0],
            key=lambda x: x["func_eval_efficiency"],
            default=None,
        )
        if best_func_result:
            best_func_evals[config_name] = best_func_result

    if best_accuracy:
        most_accurate = min(best_accuracy.items(), key=lambda x: x[1]["position_error"])
        timestep_key = (
            "initial_timestep" if "initial_timestep" in most_accurate[1] else "timestep"
        )
        print(
            f"🎯 Most Accurate: {most_accurate[0]} (pos error: {most_accurate[1]['position_error']:.2e} m)"
        )
        print(
            f"   Initial timestep: {most_accurate[1][timestep_key]:.6f}s, Tolerance: {most_accurate[1].get('tolerance', most_accurate[1].get('relative_tolerance', 'Unknown'))}"
        )
        print(
            f"   Sequence: {most_accurate[1].get('extrapolation_sequence', 'Unknown')}, Max steps: {most_accurate[1].get('max_steps', 'Unknown')}"
        )

    if best_speed:
        fastest = min(best_speed.items(), key=lambda x: x[1]["runtime"])
        timestep_key = (
            "initial_timestep" if "initial_timestep" in fastest[1] else "timestep"
        )
        print(f"⚡ Fastest: {fastest[0]} (runtime: {fastest[1]['runtime']:.4f}s)")
        print(
            f"   Initial timestep: {fastest[1][timestep_key]:.6f}s, Tolerance: {fastest[1].get('tolerance', fastest[1].get('relative_tolerance', 'Unknown'))}"
        )
        print(
            f"   Sequence: {fastest[1].get('extrapolation_sequence', 'Unknown')}, Max steps: {fastest[1].get('max_steps', 'Unknown')}"
        )

    if best_efficiency:
        most_efficient = max(best_efficiency.items(), key=lambda x: x[1]["efficiency"])
        timestep_key = (
            "initial_timestep"
            if "initial_timestep" in most_efficient[1]
            else "timestep"
        )
        print(
            f"⚖️  Most Efficient: {most_efficient[0]} (efficiency: {most_efficient[1]['efficiency']:.2e})"
        )
        print(
            f"   Initial timestep: {most_efficient[1][timestep_key]:.6f}s, Tolerance: {most_efficient[1].get('tolerance', most_efficient[1].get('relative_tolerance', 'Unknown'))}"
        )
        print(
            f"   Sequence: {most_efficient[1].get('extrapolation_sequence', 'Unknown')}, Max steps: {most_efficient[1].get('max_steps', 'Unknown')}"
        )

    if best_func_evals:
        best_func_eval = max(
            best_func_evals.items(), key=lambda x: x[1]["func_eval_efficiency"]
        )
        timestep_key = (
            "initial_timestep"
            if "initial_timestep" in best_func_eval[1]
            else "timestep"
        )
        func_evals = best_func_eval[1].get(
            "function_evaluations",
            best_func_eval[1].get("num_function_evaluations", "Unknown"),
        )
        print(
            f"🔢 Best Function Evaluation Efficiency: {best_func_eval[0]} ({func_evals} evaluations)"
        )
        print(
            f"   Initial timestep: {best_func_eval[1][timestep_key]:.6f}s, Tolerance: {best_func_eval[1].get('tolerance', best_func_eval[1].get('relative_tolerance', 'Unknown'))}"
        )
        print(
            f"   Sequence: {best_func_eval[1].get('extrapolation_sequence', 'Unknown')}, Max steps: {best_func_eval[1].get('max_steps', 'Unknown')}"
        )

    # Variable stepsize specific insights
    print("\nVARIABLE STEPSIZE PARAMETER INSIGHTS:")
    print(
        "📊 Bulirsch-Stoer sequence (2,4,6,8,12,16...): Traditional sequence, well-tested"
    )
    print("📊 Deufelhard sequence (2,4,6,8,10,12,14...): More uniform step progression")
    print(
        "📊 Lower tolerance (1e-10 to 1e-12): Better accuracy but more function evaluations"
    )
    print(
        "📊 Higher tolerance (1e-8 to 1e-6): Faster execution but may sacrifice accuracy"
    )
    print(
        "📊 Function evaluations: Critical metric for variable stepsize - shows computational cost"
    )
    print("📊 Average stepsize: Indicates how much the solver adapts during simulation")

    # Error method specific notes
    if error_method == "richardson":
        print(
            "\n📊 Note: Richardson extrapolation shows relative convergence between h and h/2 timesteps"
        )
        print(
            "📊 Variable stepsize: Average stepsize shown instead of half-step runtime"
        )
    elif error_method == "benchmark":
        print(
            "\n📊 Note: Benchmark comparison shows absolute accuracy vs high-resolution Forward Euler solution"
        )
        print("📊 Variable stepsize: Speedup calculated vs benchmark runtime")
    else:  # analytical
        print(
            "\n📊 Note: Analytical comparison shows absolute accuracy vs perfect Keplerian orbital solution"
        )
        print("📊 Variable stepsize: Efficiency considers adaptive stepping advantages")


def plot_bulirsch_stoer_parameter_comparison(all_results, save_plot=True):
    """Plot comparison between different Bulirsch-Stoer parameter combinations."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    # Create figure with multiple subplots - 2x3 layout for comprehensive analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 12))

    if error_method == "richardson":
        title_suffix = "(Richardson Extrapolation)"
    elif error_method == "benchmark":
        title_suffix = "(vs High-Resolution Benchmark)"
    else:  # analytical
        title_suffix = "(vs Analytical Keplerian Solution)"

    fig.suptitle(
        f"Bulirsch-Stoer Parameter Sweep Analysis {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (config_name, results), color in zip(all_results.items(), colors):
        if not results:  # Skip if no results
            continue

        timesteps = [r["timestep"] for r in results]
        position_errors = [r["position_error"] for r in results]
        velocity_errors = [r["velocity_error"] for r in results]
        orientation_errors = [r["orientation_error"] for r in results]
        angular_rate_errors = [r["angular_rate_error"] for r in results]
        runtimes = [r["runtime"] for r in results]

        # Plot 1: Position Errors
        ax1.loglog(
            timesteps,
            position_errors,
            "o-",
            label=config_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 2: Velocity Errors
        ax2.loglog(
            timesteps,
            velocity_errors,
            "s-",
            label=config_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 3: Orientation Errors
        ax3.loglog(
            timesteps,
            orientation_errors,
            "^-",
            label=config_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 4: Angular Rate Errors
        ax4.loglog(
            timesteps,
            angular_rate_errors,
            "v-",
            label=config_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 5: Runtime vs Timestep
        ax5.loglog(
            runtimes,
            np.array(orientation_errors),
            "d-",
            label=config_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 6: Accuracy vs Performance Trade-off
        ax6.scatter(
            timesteps,
            np.array(orientation_errors),
            label=config_name,
            color=color,
            s=60,
            alpha=0.7,
        )

    # Configure plots
    ax1.set_xlabel("Timestep (s)")
    ax1.set_ylabel(f"Position Error vs {error_method.title()} (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("Position Error vs Timestep")

    ax2.set_xlabel("Timestep (s)")
    ax2.set_ylabel(f"Velocity Error vs {error_method.title()} (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Velocity Error vs Timestep")

    ax3.set_xlabel("Timestep (s)")
    ax3.set_ylabel("Orientation Error (rad)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("Orientation Error vs Timestep")

    ax4.set_xlabel("Timestep (s)")
    ax4.set_ylabel("Angular Rate Error (rad/s)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title("Angular Rate Error vs Timestep")

    ax5.set_xlabel("Runtimes (s)")
    ax5.set_ylabel("Quaternion Error (s)")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title("Runtime vs Timestep")

    ax6.set_xlabel("Timesteps (s)")
    ax6.set_ylabel(f"Quaternion Error vs {error_method.title()} (m)")
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_title("Accuracy vs Performance Trade-off")

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bulirsch_stoer_parameter_sweep_{error_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

    plt.show()
    return fig


def plot_bulirsch_stoer_variable_step_comparison(all_results, save_plot=True):
    """Plot comparison between different Bulirsch-Stoer variable stepsize parameter combinations."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    # Create figure with multiple subplots - 2x3 layout for comprehensive analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 12))

    if error_method == "richardson":
        title_suffix = "(High-Precision Comparison)"
    elif error_method == "benchmark":
        title_suffix = "(vs High-Resolution Benchmark)"
    else:  # analytical
        title_suffix = "(vs Analytical Keplerian Solution)"

    fig.suptitle(
        f"Bulirsch-Stoer Variable Stepsize Parameter Sweep {title_suffix}",
        fontsize=16,
        fontweight="bold",
    )

    # Group configurations by base integrator (sequence + max_steps)
    # Extract base configuration (without tolerance details)
    base_configs = {}
    for config_name, results in all_results.items():
        if not results:
            continue

        # Parse config name to extract base configuration
        # e.g., "BS-VAR-bulirsch_stoer-8steps-tol1e-07-sf0.8-inc300.0"
        # Base: "bulirsch_stoer-8steps"
        parts = config_name.split("-")
        if len(parts) >= 4:
            # Extract sequence and max_steps
            sequence = parts[2]  # e.g., "bulirsch_stoer" or "deufelhard"
            max_steps = parts[3]  # e.g., "8steps"
            base_config = f"{sequence}-{max_steps}"

            if base_config not in base_configs:
                base_configs[base_config] = []
            base_configs[base_config].append((config_name, results))

    # Assign colors to base configurations
    colors = plt.cm.tab10(np.linspace(0, 1, len(base_configs)))
    base_config_colors = dict(zip(base_configs.keys(), colors))

    for base_config, config_list in base_configs.items():
        color = base_config_colors[base_config]

        # Sort configurations by tolerance (highest to lowest for better visualization)
        def extract_tolerance(config_name):
            # Extract tolerance from config name like "BS-VAR-bulirsch_stoer-8steps-tol1e-07-sf0.8-inc300.0"
            tol_part = config_name.split("tol")[1].split("-sf")[0]  # Gets "1e-07"
            return float(tol_part)  # Convert "1e-07" to 1e-07

        config_list.sort(key=lambda x: extract_tolerance(x[0]), reverse=True)

        # Collect all data points for this base configuration
        all_function_evaluations = []
        all_position_errors = []
        all_velocity_errors = []
        all_orientation_errors = []
        all_angular_rate_errors = []
        all_runtimes = []
        tolerance_labels = []

        for config_name, results in config_list:
            for r in results:
                # Extract function evaluations
                func_evals = r.get(
                    "function_evaluations", r.get("num_function_evaluations", None)
                )
                if func_evals is not None and isinstance(func_evals, (int, float)):
                    all_function_evaluations.append(func_evals)
                else:
                    # Fallback: estimate from runtime and typical evaluation rate
                    all_function_evaluations.append(r["runtime"] * 1000)

                all_position_errors.append(r["position_error"])
                all_velocity_errors.append(r["velocity_error"])
                all_orientation_errors.append(r["orientation_error"])
                all_angular_rate_errors.append(r["angular_rate_error"])
                all_runtimes.append(r["runtime"])

                # Extract tolerance for labeling
                tolerance_str = config_name.split("tol")[1].split("-")[0]
                tolerance_labels.append(f"{tolerance_str}")

        if not all_function_evaluations:
            continue

        # Create display name for legend
        display_name = (
            base_config.replace("bulirsch_stoer", "BS")
            .replace("deufelhard", "Deuf")
            .replace("steps", "s")
        )

        # Plot 1: Position Errors vs Function Evaluations
        ax1.loglog(
            all_function_evaluations,
            all_position_errors,
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

        # Plot 2: Velocity Errors vs Function Evaluations
        ax2.loglog(
            all_function_evaluations,
            all_velocity_errors,
            "s-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

        # Plot 3: Runtime vs Function Evaluations
        ax3.loglog(
            all_function_evaluations,
            all_runtimes,
            "^-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

        # Plot 4: Angular Rate Errors vs Function Evaluations
        ax4.loglog(
            all_function_evaluations,
            all_angular_rate_errors,
            "v-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

        # Plot 5: Orientation Errors vs Function Evaluations
        ax5.loglog(
            all_function_evaluations,
            all_orientation_errors,
            "d-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=6,
            alpha=0.8,
        )

        # Plot 6: Position Error vs Runtime (Efficiency)
        ax6.loglog(
            all_runtimes,
            all_position_errors,
            "o-",
            label=display_name,
            color=color,
            linewidth=2,
            markersize=8,
            alpha=0.7,
        )

    # Configure plots
    ax1.set_xlabel("Function Evaluations")
    ax1.set_ylabel(f"Position Error vs {error_method.title()} (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax1.set_title("Position Error vs Function Evaluations")

    ax2.set_xlabel("Function Evaluations")
    ax2.set_ylabel(f"Velocity Error vs {error_method.title()} (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax2.set_title("Velocity Error vs Function Evaluations")

    ax3.set_xlabel("Function Evaluations")
    ax3.set_ylabel("Runtime (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax3.set_title("Runtime vs Function Evaluations")

    ax4.set_xlabel("Function Evaluations")
    ax4.set_ylabel("Angular Rate Error (rad/s)")
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax4.set_title("Angular Rate Error vs Function Evaluations")

    ax5.set_xlabel("Function Evaluations")
    ax5.set_ylabel("Orientation Error (rad)")
    ax5.grid(True, alpha=0.3)
    ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax5.set_title("Orientation Error vs Function Evaluations")

    ax6.set_xlabel("Runtime (s)")
    ax6.set_ylabel(f"Position Error vs {error_method.title()} (m)")
    ax6.grid(True, alpha=0.3)
    ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax6.set_title("Accuracy vs Performance Trade-off")

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bulirsch_stoer_variable_step_sweep_{error_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

    plt.show()
    return fig


def plot_convergence_results(results, save_plot=True):
    """Plot the convergence analysis results with appropriate error method."""
    timesteps = [r["timestep"] for r in results]
    position_errors = [r["position_error"] for r in results]
    velocity_errors = [r["velocity_error"] for r in results]
    quaternion_errors = [r["quaternion_error"] for r in results]
    angular_velocity_errors = [r["angular_velocity_error"] for r in results]
    angular_rate_errors = [r["angular_rate_error"] for r in results]
    orientation_errors = [r["orientation_error"] for r in results]
    runtimes = [r["runtime"] for r in results]

    error_method = results[0]["error_method"]

    # Calculate speedup/overhead based on method
    if error_method == "richardson":
        reference_runtimes = [r["half_timestep_runtime"] for r in results]
        speedup_label = "Half-timestep Overhead"
        speedup_values = [ref / rt for ref, rt in zip(reference_runtimes, runtimes)]
        title_suffix = "(Richardson Extrapolation)"
    elif error_method == "benchmark":
        reference_runtime = results[0]["reference_runtime"]
        speedup_label = "Speedup vs Benchmark"
        speedup_values = [reference_runtime / rt for rt in runtimes]
        title_suffix = "(vs High-Resolution Benchmark)"
    else:  # analytical
        speedup_label = "Performance (relative)"
        speedup_values = [
            1.0 / rt for rt in runtimes
        ]  # Inverse of runtime for relative comparison
        title_suffix = "(vs Analytical Keplerian Solution)"

    # Create subplots - expanded to 2x3 for additional angular analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f"Timestep Convergence Analysis {title_suffix}", fontsize=16, fontweight="bold"
    )

    # Plot 1: Physical Errors vs Timestep
    ax1.loglog(
        timesteps,
        position_errors,
        "o-",
        label="Position Error (m)",
        linewidth=2,
        markersize=6,
    )
    ax1.loglog(
        timesteps,
        velocity_errors,
        "s-",
        label="Velocity Error (m/s)",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Timestep (s)")
    ax1.set_ylabel(f"Error vs {error_method.title()}")
    ax1.set_title(
        f"Physical State Errors vs Timestep ({error_method.title()} Comparison)"
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Quaternion-based Rotational Errors vs Timestep
    ax2.loglog(
        timesteps,
        quaternion_errors,
        "^-",
        label="Quaternion Error",
        linewidth=2,
        markersize=6,
        color="green",
    )
    ax2.loglog(
        timesteps,
        angular_velocity_errors,
        "d-",
        label="Angular Velocity Error (rad/s)",
        linewidth=2,
        markersize=6,
        color="orange",
    )
    ax2.set_xlabel("Timestep (s)")
    ax2.set_ylabel(f"Error vs {error_method.title()}")
    ax2.set_title(
        f"Quaternion & Angular Velocity Errors ({error_method.title()} Comparison)"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Angular Rate Errors vs Timestep
    ax3.loglog(
        timesteps,
        angular_rate_errors,
        "v-",
        label="Angular Rate Error (rad/s)",
        linewidth=2,
        markersize=6,
        color="purple",
    )
    ax3.set_xlabel("Timestep (s)")
    ax3.set_ylabel("Angular Rate Error (rad/s)")
    ax3.set_title(
        f"Angular Rate Errors vs Timestep ({error_method.title()} Comparison)"
    )
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Orientation Errors vs Timestep
    ax4.loglog(
        timesteps,
        orientation_errors,
        "s-",
        label="Orientation Error (rad)",
        linewidth=2,
        markersize=6,
        color="red",
    )
    ax4.set_xlabel("Timestep (s)")
    ax4.set_ylabel("Orientation Error (rad)")
    ax4.set_title(f"Orientation Errors vs Timestep ({error_method.title()} Comparison)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Runtime vs Timestep & Performance metric
    # Plot 5: Runtime vs Timestep & Performance metric
    if error_method != "analytical":
        ax5_twin = ax5.twinx()
        line1 = ax5.loglog(
            timesteps,
            runtimes,
            "o-",
            color="red",
            linewidth=2,
            markersize=6,
            label="Runtime",
        )
        line2 = ax5_twin.semilogx(
            timesteps,
            speedup_values,
            "s-",
            color="blue",
            linewidth=2,
            markersize=6,
            label=speedup_label,
        )
        ax5.set_xlabel("Timestep (s)")
        ax5.set_ylabel("Runtime (s)", color="red")
        ax5_twin.set_ylabel(speedup_label, color="blue")
        ax5.set_title("Runtime & Performance vs Timestep")
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis="y", labelcolor="red")
        ax5_twin.tick_params(axis="y", labelcolor="blue")

        # Add legend for twin axis plot
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax5.legend(lines, labels, loc="upper right")
    else:
        # For analytical, just show runtime
        ax5.loglog(timesteps, runtimes, "o-", color="red", linewidth=2, markersize=6)
        ax5.set_xlabel("Timestep (s)")
        ax5.set_ylabel("Runtime (s)")
        ax5.set_title("Runtime vs Timestep")
        ax5.grid(True, alpha=0.3)

    # Plot 6: Accuracy vs Performance Trade-off
    ax6.loglog(
        runtimes,
        quaternion_errors,
        "o-",
        label="Position Error",
        linewidth=2,
        markersize=6,
    )
    # ax6.loglog(runtimes, velocity_errors, 's-', label='Velocity Error', linewidth=2, markersize=6)
    ax6.set_xlabel("Runtime (s)")
    ax6.set_ylabel(f"Error vs {error_method.title()}")
    ax6.set_title("Quaternion Error vs Performance Trade-off")
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timestep_convergence_{error_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

    plt.show()

    return fig


def print_convergence_summary(results):
    """Print a summary table of the convergence analysis with appropriate error method."""
    error_method = results[0]["error_method"]

    print("\n" + "=" * 160)
    print(f"CONVERGENCE SUMMARY (vs {error_method.upper()})")
    print("=" * 160)

    if error_method == "richardson":
        header = f"{'Timestep (s)':<12} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'AngRate Err (rad/s)':<18} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Half-step RT':<12}"
        print(header)
        print("-" * 160)

        for r in results:
            half_rt = r.get("half_timestep_runtime", 0)
            row = f"{r['timestep']:<12.6f} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['angular_rate_error']:<18.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {half_rt:<12.4f}"
            print(row)

        print("\n" + "=" * 160)
        print("RECOMMENDATIONS (Richardson Extrapolation)")
        print("=" * 160)
        print("Note: Richardson errors are relative between h and h/2 timesteps")

        # Position-based recommendations for Richardson
        print("\nPOSITION ACCURACY RECOMMENDATIONS:")
        for r in results:
            if r["position_error"] < 1e-6:  # Sub-micrometer position accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"✅ For high precision (< 1µm position error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        for r in results:
            if r["position_error"] < 1e-3:  # Millimeter position accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"⚠️  For moderate precision (< 1mm position error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        for r in results:
            if r["position_error"] < 1e-1:  # Centimeter position accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"🔥 For low precision (< 1cm position error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        # Angular accuracy recommendations for Richardson
        print("\nANGULAR ACCURACY RECOMMENDATIONS:")
        for r in results:
            if r["orientation_error"] < 1e-6:  # Sub-microradian orientation accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"✅ For high angular precision (< 1µrad orientation error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        for r in results:
            if r["orientation_error"] < 1e-3:  # Milliradian orientation accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"⚠️  For moderate angular precision (< 1mrad orientation error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        for r in results:
            if r["angular_rate_error"] < 1e-3:  # Milliradian/s angular rate accuracy
                half_rt = r.get("half_timestep_runtime", 0)
                overhead = half_rt / r["runtime"] if r["runtime"] > 0 else 0
                print(
                    f"🌀 For angular rate precision (< 1mrad/s rate error): timestep ≤ {r['timestep']:.6f}s ({overhead:.1f}x overhead)"
                )
                break

        print(
            "\n📊 Richardson extrapolation: Half-timestep runtime overhead measures convergence validation cost"
        )

    elif error_method == "benchmark":
        benchmark_runtime = results[0]["reference_runtime"]

        header = f"{'Timestep (s)':<12} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'AngRate Err (rad/s)':<18} {'Orient Err (rad)':<15} {'Runtime (s)':<12} {'Speedup':<8}"
        print(header)
        print("-" * 160)

        for r in results:
            speedup = benchmark_runtime / r["runtime"]
            row = f"{r['timestep']:<12.6f} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['angular_rate_error']:<18.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f} {speedup:<8.1f}x"
            print(row)

        print("\n" + "=" * 160)
        print("RECOMMENDATIONS (vs High-Resolution Benchmark)")
        print("=" * 160)

        # Position-based recommendations
        print("POSITION ACCURACY RECOMMENDATIONS:")
        for r in results:
            if r["position_error"] < 1e-6:  # Sub-micrometer position accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"✅ For high precision (< 1µm position error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

        for r in results:
            if r["position_error"] < 1e-3:  # Millimeter position accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"⚠️  For moderate precision (< 1mm position error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

        for r in results:
            if r["position_error"] < 1e-1:  # Centimeter position accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"🔥 For low precision (< 1cm position error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

        print(f"\n📊 Benchmark runtime (Forward Euler, 1ms): {benchmark_runtime:.2f}s")

        # Angular accuracy recommendations for benchmark
        print("\nANGULAR ACCURACY RECOMMENDATIONS:")
        for r in results:
            if r["orientation_error"] < 1e-6:  # Sub-microradian orientation accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"✅ For high angular precision (< 1µrad orientation error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

        for r in results:
            if r["orientation_error"] < 1e-3:  # Milliradian orientation accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"⚠️  For moderate angular precision (< 1mrad orientation error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

        for r in results:
            if r["angular_rate_error"] < 1e-3:  # Milliradian/s angular rate accuracy
                speedup = benchmark_runtime / r["runtime"]
                print(
                    f"🌀 For angular rate precision (< 1mrad/s rate error): timestep ≤ {r['timestep']:.6f}s ({speedup:.1f}x speedup)"
                )
                break

    else:  # analytical
        header = f"{'Timestep (s)':<12} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'AngRate Err (rad/s)':<18} {'Orient Err (rad)':<15} {'Runtime (s)':<12}"
        print(header)
        print("-" * 145)

        for r in results:
            row = f"{r['timestep']:<12.6f} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['angular_rate_error']:<18.2e} {r['orientation_error']:<15.2e} {r['runtime']:<12.4f}"
            print(row)

        print("\n" + "=" * 160)
        print("RECOMMENDATIONS (vs Analytical Keplerian Solution)")
        print("=" * 160)

        # Position-based recommendations for analytical
        print("POSITION ACCURACY RECOMMENDATIONS:")
        for r in results:
            if r["position_error"] < 1e-6:  # Sub-micrometer position accuracy
                print(
                    f"✅ For high precision (< 1µm position error): timestep ≤ {r['timestep']:.6f}s"
                )
                break

        for r in results:
            if r["position_error"] < 1e-3:  # Millimeter position accuracy
                print(
                    f"⚠️  For moderate precision (< 1mm position error): timestep ≤ {r['timestep']:.6f}s"
                )
                break

        for r in results:
            if r["position_error"] < 1e-1:  # Centimeter position accuracy
                print(
                    f"🔥 For low precision (< 1cm position error): timestep ≤ {r['timestep']:.6f}s"
                )
                break

        # Angular accuracy recommendations for analytical (note: analytical has zero angular errors)
        print("\nNOTE: Analytical Keplerian solution only covers translational motion.")
        print(
            "Angular errors are zero since rotation is not included in analytical reference."
        )

        print("\n📊 Analytical solution provides perfect Keplerian reference")

    # Common recommendations for all methods
    print("\nCOMMON RECOMMENDATIONS:")

    # Runtime efficiency
    fastest = min(results, key=lambda x: x["runtime"])
    print(
        f"⚡ Fastest simulation: timestep = {fastest['timestep']:.6f}s, runtime = {fastest['runtime']:.4f}s"
    )

    # Most accurate overall (position + angular combined)
    most_accurate = min(
        results,
        key=lambda x: x["position_error"]
        + x["orientation_error"]
        + x["angular_rate_error"],
    )
    print(f"🎯 Most accurate overall: timestep = {most_accurate['timestep']:.6f}s")
    print(f"   Position error: {most_accurate['position_error']:.2e} m")
    print(f"   Orientation error: {most_accurate['orientation_error']:.2e} rad")
    print(f"   Angular rate error: {most_accurate['angular_rate_error']:.2e} rad/s")
    print(f"   Runtime: {most_accurate['runtime']:.4f}s")

    # Best compromise based on multiple criteria including angular metrics
    best_compromise = None
    for r in results:
        if (
            r["position_error"] < 1e-3
            and r["orientation_error"] < 1e-3
            and r["angular_rate_error"] < 1e-3
            and (best_compromise is None or r["runtime"] < best_compromise["runtime"])
        ):
            best_compromise = r

    if best_compromise:
        print(
            f"⚖️  Recommended compromise (pos < 1mm, orient < 1e-3 rad, angular rate < 1e-3 rad/s): timestep = {best_compromise['timestep']:.6f}s"
        )
        print(f"   Position error: {best_compromise['position_error']:.2e} m")
        print(f"   Orientation error: {best_compromise['orientation_error']:.2e} rad")
        print(
            f"   Angular rate error: {best_compromise['angular_rate_error']:.2e} rad/s"
        )
        print(f"   Runtime: {best_compromise['runtime']:.4f}s")

    # Best angular accuracy (for attitude-critical applications)
    best_angular = min(
        results, key=lambda x: max(x["orientation_error"], x["angular_rate_error"])
    )
    print(f"🌀 Best angular accuracy: timestep = {best_angular['timestep']:.6f}s")
    print(f"   Orientation error: {best_angular['orientation_error']:.2e} rad")
    print(f"   Angular rate error: {best_angular['angular_rate_error']:.2e} rad/s")
    print(f"   Runtime: {best_angular['runtime']:.4f}s")

    # Application-specific recommendations
    print("\nAPPLICATION-SPECIFIC RECOMMENDATIONS:")
    print(
        "🛰️  For attitude control systems: prioritize orientation & angular rate accuracy"
    )
    print("🚀 For trajectory planning: prioritize position & velocity accuracy")
    print("⚡ For real-time applications: prioritize runtime performance")
    print("🎯 For scientific analysis: prioritize overall accuracy")


def plot_integrator_comparison(all_results, save_plot=True):
    """Plot comparison between different integrators with appropriate error method."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    # Create figure with multiple subplots - expanded to 2x3 for additional angular analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(22, 12))

    if error_method == "richardson":
        title_suffix = "(Richardson Extrapolation)"
    elif error_method == "benchmark":
        title_suffix = "(vs High-Resolution Benchmark)"
    else:  # analytical
        title_suffix = "(vs Analytical Keplerian Solution)"

    fig.suptitle(
        f"Integrator Comparison Analysis {title_suffix}", fontsize=16, fontweight="bold"
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (integrator_name, results), color in zip(all_results.items(), colors):
        if not results:  # Skip if no results
            continue

        timesteps = [r["timestep"] for r in results]
        position_errors = [r["position_error"] for r in results]
        velocity_errors = [r["velocity_error"] for r in results]
        angular_rate_errors = [r["angular_rate_error"] for r in results]
        orientation_errors = [r["orientation_error"] for r in results]
        runtimes = [r["runtime"] for r in results]

        # Plot 1: Position Errors
        ax1.loglog(
            timesteps,
            position_errors,
            "o-",
            label=integrator_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 2: Velocity Errors
        ax2.loglog(
            timesteps,
            velocity_errors,
            "s-",
            label=integrator_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 3: Angular Rate Errors
        ax3.loglog(
            timesteps,
            angular_rate_errors,
            "v-",
            label=integrator_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 4: Orientation Errors
        ax4.loglog(
            timesteps,
            orientation_errors,
            "^-",
            label=integrator_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 5: Runtime vs Timestep
        ax5.loglog(
            timesteps,
            runtimes,
            "d-",
            label=integrator_name,
            color=color,
            linewidth=2,
            markersize=6,
        )

        # Plot 6: Accuracy vs Performance Trade-off
        ax6.scatter(
            timesteps,
            np.array(orientation_errors),
            label=integrator_name,
            color=color,
            s=60,
            alpha=0.7,
        )

    # Configure plots
    ax1.set_xlabel("Timestep (s)")
    ax1.set_ylabel(f"Position Error vs {error_method.title()} (m)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("Position Error vs Timestep")

    ax2.set_xlabel("Timestep (s)")
    ax2.set_ylabel(f"Velocity Error vs {error_method.title()} (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("Velocity Error vs Timestep")

    ax3.set_xlabel("Timestep (s)")
    ax3.set_ylabel("Angular Rate Error (rad/s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("Angular Rate Error vs Timestep")

    ax4.set_xlabel("Timestep (s)")
    ax4.set_ylabel("Orientation Error (rad)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_title("Orientation Error vs Timestep")

    ax5.set_xlabel("Timestep (s)")
    ax5.set_ylabel("Runtime (s)")
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_title("Runtime vs Timestep")

    ax6.set_xlabel("timestep (s)")
    ax6.set_ylabel(f"Quaternion Error vs {error_method.title()} (m)")
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    ax6.set_title("Quaternion accuracy vs Performance Trade-off")

    plt.tight_layout()

    if save_plot:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"integrator_comparison_{error_method}_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved as: {filename}")

    plt.show()
    return fig


def print_integrator_comparison_summary(all_results):
    """Print a summary table comparing different integrators with appropriate error method."""

    # Determine error method from first result
    error_method = "richardson"  # default
    for results in all_results.values():
        if results and "error_method" in results[0]:
            error_method = results[0]["error_method"]
            break

    print("\n" + "=" * 120)
    print(f"INTEGRATOR COMPARISON SUMMARY (vs {error_method.upper()})")
    print("=" * 120)

    # For each timestep, compare integrators
    if not all_results:
        print("No results to display")
        return

    # Get all timesteps that were tested
    all_timesteps = set()
    for results in all_results.values():
        for r in results:
            all_timesteps.add(r["timestep"])
    all_timesteps = sorted(all_timesteps, reverse=True)

    for timestep in all_timesteps:
        print(f"\n--- TIMESTEP: {timestep:.6f}s ---")

        if error_method == "richardson":
            header = f"{'Integrator':<20} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Runtime (s)':<12} {'Half-step RT (s)':<15}"
        elif error_method == "benchmark":
            header = f"{'Integrator':<20} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Runtime (s)':<12} {'Speedup':<12}"
        else:  # analytical
            header = f"{'Integrator':<20} {'Pos Error (m)':<15} {'Vel Error (m/s)':<15} {'Runtime (s)':<12} {'Efficiency':<12}"

        print(header)
        print("-" * 94)

        timestep_results = []
        for int_name, results in all_results.items():
            for r in results:
                if abs(r["timestep"] - timestep) < 1e-10:  # Match timestep
                    if error_method == "benchmark":
                        benchmark_runtime = r.get(
                            "reference_runtime", r.get("benchmark_runtime", 1.0)
                        )
                        metric = benchmark_runtime / r["runtime"]  # speedup
                    elif error_method == "richardson":
                        metric = r.get(
                            "half_timestep_runtime", r.get("reference_runtime", 0)
                        )  # half-step runtime
                    else:  # analytical
                        efficiency = (
                            (1.0 / r["position_error"]) / r["runtime"]
                            if r["position_error"] > 0
                            else 0
                        )
                        metric = efficiency

                    timestep_results.append(
                        {
                            "integrator": int_name,
                            "position_error": r["position_error"],
                            "velocity_error": r["velocity_error"],
                            "runtime": r["runtime"],
                            "metric": metric,
                        }
                    )
                    break

        # Sort by position error (most accurate first)
        timestep_results.sort(key=lambda x: x["position_error"])

        for i, r in enumerate(timestep_results):
            marker = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "

            if error_method == "benchmark":
                row = f"{marker} {r['integrator']:<17} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<12.1f}x"
            elif error_method == "richardson":
                row = f"{marker} {r['integrator']:<17} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<15.4f}"
            else:  # analytical
                row = f"{marker} {r['integrator']:<17} {r['position_error']:<15.2e} {r['velocity_error']:<15.2e} {r['runtime']:<12.4f} {r['metric']:<12.2e}"

            print(row)

    # Overall recommendations
    print("\n" + "=" * 120)
    print(f"OVERALL RECOMMENDATIONS (vs {error_method.upper()})")
    print("=" * 120)

    # Find best integrator for different criteria
    best_accuracy = {}
    best_speed = {}
    best_efficiency = {}

    for int_name, results in all_results.items():
        if not results:
            continue

        # Best accuracy (smallest position error)
        best_acc_result = min(results, key=lambda x: x["position_error"])
        best_accuracy[int_name] = best_acc_result

        # Best speed (fastest runtime at any timestep)
        best_speed_result = min(results, key=lambda x: x["runtime"])
        best_speed[int_name] = best_speed_result

        # Best efficiency (accuracy/runtime ratio)
        for r in results:
            r["efficiency"] = (
                (1.0 / r["position_error"]) / r["runtime"]
                if r["position_error"] > 0
                else 0
            )
        best_eff_result = max(results, key=lambda x: x["efficiency"])
        best_efficiency[int_name] = best_eff_result

    if best_accuracy:
        most_accurate = min(best_accuracy.items(), key=lambda x: x[1]["position_error"])
        print(
            f"🎯 Most Accurate: {most_accurate[0]} (pos error: {most_accurate[1]['position_error']:.2e} m at timestep {most_accurate[1]['timestep']:.6f}s)"
        )

    if best_speed:
        fastest = min(best_speed.items(), key=lambda x: x[1]["runtime"])
        print(
            f"⚡ Fastest: {fastest[0]} (runtime: {fastest[1]['runtime']:.4f}s at timestep {fastest[1]['timestep']:.6f}s)"
        )

    if best_efficiency:
        most_efficient = max(best_efficiency.items(), key=lambda x: x[1]["efficiency"])
        print(
            f"⚖️  Most Efficient: {most_efficient[0]} (efficiency: {most_efficient[1]['efficiency']:.2e} at timestep {most_efficient[1]['timestep']:.6f}s)"
        )

    # Error method specific notes
    if error_method == "richardson":
        print(
            "\n📊 Note: Richardson extrapolation shows relative convergence between h and h/2 timesteps"
        )
    elif error_method == "benchmark":
        print(
            "\n📊 Note: Benchmark comparison shows absolute accuracy vs high-resolution Forward Euler solution"
        )
    else:  # analytical
        print(
            "\n📊 Note: Analytical comparison shows absolute accuracy vs perfect Keplerian orbital solution"
        )


def main():
    print("=" * 60)
    print("BioInspired - Timestep Convergence Analyzer")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Ask user what analysis to run
    print("\nSelect analysis type:")
    print("1. Single integrator convergence analysis")
    print("2. Runge-Kutta coefficient sets comparison")
    print("3. Integrator types comparison (RK vs ABM vs Bulirsch-Stoer)")
    print("4. Combined analysis (RK coefficient sets + integrator types)")
    print("5. Bulirsch-Stoer parameter sweep optimization")
    print("6. Bulirsch-Stoer variable stepsize parameter sweep")
    print("7. Monte Carlo sensitivity analysis")
    choice = input("Enter choice (1-7): ").strip()

    # Ask for error estimation method
    print("\nSelect error estimation method:")
    print("1. Richardson extrapolation (compare with half-timestep)")
    print("2. High-resolution benchmark (Forward Euler 1ms)")
    print("3. Analytical Keplerian solution (perfect reference)")
    error_choice = input("Enter error method (1-3): ").strip()

    error_method_map = {"1": "richardson", "2": "benchmark", "3": "analytical"}
    error_method = error_method_map.get(error_choice, "richardson")

    # timesteps = np.logspace(np.log10(0.1), np.log10(0.5), num=5, base=10)  # Default timesteps from 0.01s up to 5s
    upper_limit = 0.5  # Upper limit for timesteps
    lower_limit = 0.1  # Lower limit for timesteps
    spacing = 0.02  # Spacing between timesteps
    # Generate n timesteps linearly spaced between lower and upper limits
    timesteps = np.linspace(
        lower_limit, upper_limit, num=round((upper_limit - lower_limit) / spacing + 1)
    )
    timesteps = timesteps[
        ::-1
    ]  # Reverse to go from large to small timesteps for better analysis progression
    # Concatenate the two arrays for timesteps
    # timesteps = np.concatenate([timesteps, np.linspace(8, 300, num=30)])
    if choice == "1":
        # Single integrator analysis
        print("\nSelect integrator type:")
        print("1. Runge-Kutta")
        print("2. Adams-Bashforth-Moulton")
        print("3. Bulirsch-Stoer")
        int_choice = input("Enter integrator type (1-3): ").strip()

        integrator_type_map = {
            "1": "runge_kutta",
            "2": "adams_bashforth_moulton",
            "3": "bulirsch_stoer",
        }

        integrator_type = integrator_type_map.get(int_choice, "runge_kutta")
        print(
            f"\nRunning single integrator analysis with {integrator_type} using {error_method} error estimation..."
        )
        results = analyze_timestep_convergence(
            timesteps,
            simulation_time=150,
            integrator_type=integrator_type,
            error_method=error_method,
        )
        print_convergence_summary(results)
        plot_convergence_results(results, save_plot=True)

    elif choice == "2":
        # Runge-Kutta coefficient sets comparison
        print(
            f"\nRunning Runge-Kutta coefficient sets comparison using {error_method} error estimation..."
        )
        all_results = analyze_integrator_comparison(
            timesteps, simulation_time=150, error_method=error_method
        )
        print_integrator_comparison_summary(all_results)
        plot_integrator_comparison(all_results, save_plot=True)

    elif choice == "3":
        # Integrator types comparison
        print(
            f"\nRunning integrator types comparison using {error_method} error estimation..."
        )
        all_results = analyze_integrator_types_comparison(
            timesteps, simulation_time=150, error_method=error_method
        )
        print_integrator_comparison_summary(all_results)
        plot_integrator_comparison(all_results, save_plot=True)

    elif choice == "4":
        # Combined analysis: RK coefficient sets + integrator types
        print(
            f"\nRunning combined RK coefficient sets and integrator types analysis using {error_method} error estimation..."
        )

        # Get RK coefficient sets results
        print("\n" + "=" * 60)
        print("PHASE 1: RUNGE-KUTTA COEFFICIENT SETS")
        print("=" * 60)
        rk_results = analyze_integrator_comparison(
            timesteps, simulation_time=150, error_method=error_method
        )

        # Get integrator types results
        print("\n" + "=" * 60)
        print("PHASE 2: INTEGRATOR TYPES COMPARISON")
        print("=" * 60)
        types_results = analyze_integrator_types_comparison(
            timesteps, simulation_time=150, error_method=error_method
        )

        # Combine results
        all_results = {**rk_results, **types_results}

        print_integrator_comparison_summary(all_results)
        plot_integrator_comparison(all_results, save_plot=True)

    elif choice == "5":
        # Bulirsch-Stoer parameter sweep optimization
        print(
            f"\nRunning Bulirsch-Stoer parameter sweep optimization using {error_method} error estimation..."
        )
        bs_results = analyze_bulirsch_stoer_parameters(
            timesteps, simulation_time=150, error_method=error_method
        )
        print_bulirsch_stoer_parameter_summary(bs_results)
        plot_bulirsch_stoer_parameter_comparison(bs_results, save_plot=True)

    elif choice == "6":
        # Bulirsch-Stoer variable stepsize parameter sweep
        print(
            f"\nRunning Bulirsch-Stoer variable stepsize parameter sweep using {error_method} error estimation..."
        )

        # Ask for initial timesteps
        print("\nInitial timestep configuration:")
        use_default = (
            input("Use default initial timesteps [10, 5, 1, 0.5, 0.1]s? (y/n): ")
            .strip()
            .lower()
        )

        if use_default == "n":
            timestep_input = input(
                "Enter initial timesteps (comma-separated, e.g., 10,5,1,0.5): "
            ).strip()
            try:
                initial_timesteps = np.array(
                    [float(x.strip()) for x in timestep_input.split(",")]
                )
            except ValueError:
                print("Invalid timesteps, using defaults")
                initial_timesteps = np.array([0.1])
        else:
            initial_timesteps = np.array([0.1])

        print(f"Using initial timesteps: {initial_timesteps}")

        bs_var_results = analyze_bulirsch_stoer_variable_step_parameters(
            initial_timesteps, simulation_time=150, error_method=error_method
        )
        print_bulirsch_stoer_variable_step_summary(
            bs_var_results
        )  # Use variable stepsize specific summary function
        plot_bulirsch_stoer_variable_step_comparison(bs_var_results, save_plot=True)

    elif choice == "7":
        # Monte Carlo sensitivity analysis
        print("\nMonte Carlo sensitivity analysis configuration:")

        # Ask for fixed timestep
        timestep_input = input("Enter fixed timestep (e.g., 0.1): ").strip()
        try:
            fixed_timestep = float(timestep_input)
        except ValueError:
            print("Invalid timestep, using default 0.1s")
            fixed_timestep = 0.1

        # Ask for integrator type
        print("\nSelect integrator type for Monte Carlo:")
        print("1. Runge-Kutta")
        print("2. Adams-Bashforth-Moulton")
        print("3. Bulirsch-Stoer")
        int_choice = input("Enter integrator type (1-3): ").strip()

        integrator_type_map = {
            "1": "runge_kutta",
            "2": "adams_bashforth_moulton",
            "3": "bulirsch_stoer",
        }
        integrator_type = integrator_type_map.get(int_choice, "runge_kutta")

        # For Runge-Kutta, ask for coefficient set
        coefficient_set = None
        if integrator_type == "runge_kutta":
            print("\nSelect Runge-Kutta coefficient set:")
            print("1. rk_4 (4th order)")
            print("2. rkf_56 (5th order)")
            print("3. rkf_78 (6th order)")
            print("4. rk_89 (7th order)")
            rk_choice = input("Enter coefficient set (1-4): ").strip()

            rk_coefficient_map = {
                "1": integrator.CoefficientSets.rk_4,
                "2": integrator.CoefficientSets.rkf_56,
                "3": integrator.CoefficientSets.rkf_78,
                "4": integrator.CoefficientSets.rkf_89,
            }
            coefficient_set = rk_coefficient_map.get(
                rk_choice, integrator.CoefficientSets.rk_4
            )

        # Ask for number of Monte Carlo samples
        samples_input = input(
            "Enter number of Monte Carlo samples (default 100): "
        ).strip()
        try:
            n_samples = int(samples_input) if samples_input else 100
        except ValueError:
            print("Invalid number of samples, using default 100")
            n_samples = 100

        # Ask for simulation time
        sim_time_input = input(
            "Enter simulation time in seconds (default 50.0): "
        ).strip()
        try:
            simulation_time = float(sim_time_input) if sim_time_input else 50.0
        except ValueError:
            print("Invalid simulation time, using default 50.0s")
            simulation_time = 50.0

        print(
            f"\nRunning Monte Carlo sensitivity analysis using {error_method} error estimation..."
        )
        print(
            f"Configuration: {integrator_type} @ {fixed_timestep:.6f}s, {n_samples} samples, {simulation_time:.1f}s simulation"
        )

        # Run Monte Carlo analysis
        if integrator_type == "runge_kutta":
            mc_results = analyze_monte_carlo_sensitivity(
                timestep=fixed_timestep,
                integrator_type=integrator_type,
                coefficient_set=coefficient_set,
                simulation_time=simulation_time,
                n_samples=n_samples,
                error_method=error_method,
            )
        else:
            mc_results = analyze_monte_carlo_sensitivity(
                timestep=fixed_timestep,
                integrator_type=integrator_type,
                simulation_time=simulation_time,
                n_samples=n_samples,
                error_method=error_method,
            )

        # Print summary and plot results
        print_monte_carlo_summary(mc_results)
        plot_monte_carlo_results(mc_results, save_plot=True)

    else:
        print(
            f"Invalid choice. Please select 1-7. Running default single integrator analysis with {error_method} error estimation..."
        )
        results = analyze_timestep_convergence(
            timesteps, simulation_time=150, error_method=error_method
        )
        print_convergence_summary(results)
        plot_convergence_results(results, save_plot=True)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Error estimation method used: {error_method.upper()}")

    if error_method == "richardson":
        print("📊 Richardson extrapolation provides relative error estimates")
    elif error_method == "benchmark":
        print(
            "📊 Benchmark comparison provides absolute error vs high-resolution solution"
        )
    else:
        print(
            "📊 Analytical comparison provides absolute error vs perfect Keplerian solution"
        )


if __name__ == "__main__":
    main()
