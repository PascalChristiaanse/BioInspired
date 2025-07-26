"""Endurance Ephemeris Playground
This module provides a playground for simulating the Endurance spacecraft to generate ephemeris data.
Includes interpolation error analysis for ephemeris accuracy assessment.
"""

import numpy as np
from typing import Dict, Any

from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.spacecraft import Endurance
from bioinspired.error_designer.InterpolationErrorDesigner import (
    InterpolationErrorDesigner,
    InterpolationType,
    InterpolationDataType,
)
from bioinspired.error_designer.ErrorDesignerBase import (
    SweepConfig,
    ParameterSweepConfig,
    SweepMode,
)

from bioinspired.error_designer.InteractivePlotter import (
    create_advanced_interactive_plot,
)

# Import tudatpy for interpolation types
try:
    import tudatpy
except ImportError:
    tudatpy = None


class EnduranceInterpolationErrorDesigner(InterpolationErrorDesigner):
    """Specialized interpolation error designer for Endurance spacecraft ephemeris analysis.

    This class analyzes the accuracy of interpolating Endurance spacecraft trajectories
    for ephemeris generation, comparing different interpolation methods and parameters.
    """

    def __init__(
        self,
        simulation_duration: float = 100.0,
        initial_state: np.ndarray = None,
        interpolation_type: InterpolationType = InterpolationType.LAGRANGE,
        data_type: InterpolationDataType = InterpolationDataType.VECTOR,
    ):
        """Initialize the Endurance interpolation error designer.

        Args:
            simulation_duration: Total simulation time for benchmark solution
            initial_state: Initial state vector for Endurance spacecraft
            interpolation_type: Type of interpolation to analyze
            data_type: Type of data being interpolated
        """
        super().__init__(interpolation_type, data_type)

        self.simulation_duration = simulation_duration
        self.initial_state = initial_state or np.array(
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 7]
        )

        # Cache for simulation components
        self._simulator = None
        self._spacecraft = None

    def _get_simulator_and_spacecraft(self):
        """Get or create simulator and spacecraft instances."""
        if self._simulator is None or self._spacecraft is None:
            self._simulator = EmptyUniverseSimulator()
            self._spacecraft = Endurance(
                simulation=self._simulator,
                initial_state=self.initial_state.copy(),
            )
        return self._simulator, self._spacecraft

    def _create_benchmark_solution(self) -> Dict[float, np.ndarray]:
        """Create high-fidelity benchmark solution using fine timesteps.

        Returns:
            Dictionary of benchmark state history {time: state_vector}
        """
        print("Creating high-fidelity benchmark solution...")

        simulator, spacecraft = self._get_simulator_and_spacecraft()

        # Reset spacecraft to initial state
        spacecraft.initial_state = self.initial_state.copy()

        # Run high-fidelity simulation (small timesteps for accuracy)
        dynamics_simulator = simulator.run(0, self.simulation_duration)
        benchmark_history = dynamics_simulator.state_history

        print(f"Benchmark solution created with {len(benchmark_history)} time points")
        return benchmark_history

    def _propagate_trajectory(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Dict[float, np.ndarray]]:
        """Propagate trajectory between two time points for Chebyshev node generation.

        Args:
            parameters: Dictionary containing:
                - 'start_epoch': Start time for propagation
                - 'end_epoch': End time for propagation
                - 'initial_state': Initial state vector at start_epoch

        Returns:
            Dictionary containing 'state_history': {time: state_vector}
        """
        start_epoch = parameters["start_epoch"]
        end_epoch = parameters["end_epoch"]
        initial_state = parameters["initial_state"]

        # Create fresh simulator and spacecraft for this propagation
        simulator = EmptyUniverseSimulator()
        _ = Endurance(  # Spacecraft is configured within simulator
            simulation=simulator,
            initial_state=initial_state.copy(),
        )

        # Propagate from start to end epoch
        dynamics_simulator = simulator.run(start_epoch, end_epoch)

        return {"state_history": dynamics_simulator.state_history}


def main():
    """Demonstrate 3D parameter sweep analysis."""
    print("Starting 3-dimensional parameter sweep demonstration...")

    # Initialize error designer
    designer = EnduranceInterpolationErrorDesigner()

    # Define 3D parameter space
    n_data_points_values = np.logspace(5, 10, num=6, dtype=int, base=2)
    interpolation_orders = [4, 6, 8]
    interpolation_types = [
        InterpolationType.LINEAR,
        InterpolationType.CUBIC_SPLINE,
        InterpolationType.LAGRANGE,
    ]

    # Create 3D parameter configuration
    config_3d = SweepConfig(
        parameters=[
            ParameterSweepConfig(
                parameter_name="n_data_points",
                parameter_values=n_data_points_values,
                parameter_display_name="Data Points",
                parameter_units="points",
            ),
            # ParameterSweepConfig(
            #     parameter_name="interpolation_type",
            #     parameter_values=interpolation_types,
            #     parameter_display_name="Interpolation Type",
            #     parameter_units="method",
            # ),
            ParameterSweepConfig(
                parameter_name="interpolation_order",
                parameter_values=interpolation_orders,
                parameter_display_name="Interpolation Order",
                parameter_units="order",
            ),
        ],
        sweep_mode=SweepMode.CARTESIAN,
    )

    print(f"3D Parameter space: {config_3d.n_dimensions} dimensions")
    print(f"Total combinations: {config_3d.total_combinations()}")
    print(
        f"Parameter space size: {len(n_data_points_values)} × {len(interpolation_types)} × {len(interpolation_orders)}"
    )

    # Run the 3D analysis
    result = designer.perform_parameter_sweep(config_3d)

    # Show summary statistics
    print("\n" + "=" * 80)
    print("3D PARAMETER SWEEP RESULTS SUMMARY")
    print("=" * 80)

    # Display timing statistics
    timing_stats = {
        "mean": np.mean(result.computation_times),
        "std": np.std(result.computation_times),
        "min": np.min(result.computation_times),
        "max": np.max(result.computation_times),
        "total": np.sum(result.computation_times),
    }

    print("Computation Time Statistics:")
    print(f"  Total runtime: {timing_stats['total']:.2f}s")
    print(
        f"  Mean per sample: {timing_stats['mean']:.4f}s ± {timing_stats['std']:.4f}s"
    )
    print(f"  Range: {timing_stats['min']:.4f}s to {timing_stats['max']:.4f}s")

    # Show error metric summary
    for error_type in result.error_metrics:
        print(f"\n{error_type.upper()} Error Analysis:")
        error_values = []
        for combo_key, time_series in result.error_metrics[error_type].items():
            mean_error = np.mean(list(time_series.values()))
            error_values.append(mean_error)

        error_stats = {
            "mean": np.mean(error_values),
            "std": np.std(error_values),
            "min": np.min(error_values),
            "max": np.max(error_values),
        }

        print(f"  Mean error: {error_stats['mean']:.2e} ± {error_stats['std']:.2e}")
        print(f"  Range: {error_stats['min']:.2e} to {error_stats['max']:.2e}")

    # Create advanced interactive plot with window splitter for 3D analysis
    print(
        "\nCreating advanced interactive 3D parameter sweep plot with window splitter..."
    )
    print("This will open a professional interface with:")
    print("  • Blender-style window splitting for multiple simultaneous views")
    print("  • Multiple plot types: Line plots, heatmaps, time series, statistics")
    print("  • Interactive controls for error types and summary statistics")
    print("  • Professional dark theme interface")
    print("  • Split panes using ⬌/⬍ buttons, drag to resize, delete by shrinking")

    try:
        create_advanced_interactive_plot(
            result,
        )
        print("Advanced interactive plot created successfully!")
        print(
            "Use the pane controls to create multiple views of your 3D parameter sweep data."
        )
        print("Try different combinations like:")
        print("  • Line plot + Statistics table + Performance plot")
        print("  • Error comparison + Time series + Heatmap")
        print("  • Multiple line plots with different error types")
    except Exception as e:
        print(f"Error creating advanced interactive plot: {e}")
        print("Continuing without interactive visualization...")
        import traceback

        traceback.print_exc()

    return result


if __name__ == "__main__":
    # Choose which analysis to run
    print("\n" + "=" * 60)
    print("RUNNING MULTIVARIATE (2D) SWEEP")
    print("=" * 60)
    main()
