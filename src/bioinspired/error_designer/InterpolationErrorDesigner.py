"""Interpolation Error Designer
Specialized error designer for analyzing interpolation accuracy against benchmark solutions.
Supports scalar, vector, and matrix interpolation using TudatPy interpolators with parameter sweeps
for data points, interpolation method, and interpolation order.
"""

import numpy as np
import time
from abc import abstractmethod
from numpy.polynomial.chebyshev import chebpts2
from typing import Dict, Any, Optional, Union
from enum import Enum

from .ErrorDesignerBase import (
    ErrorDesignerBase,
    ErrorMethod,
)

try:
    from tudatpy.math import interpolators

    TUDAT_AVAILABLE = True
except ImportError:
    TUDAT_AVAILABLE = False
    print("Warning: TudatPy not available. Interpolation analysis will be limited.")


class InterpolationType(Enum):
    """Enumeration of supported interpolation types."""

    LINEAR = "linear"
    PIECEWISE_CONSTANT = "piecewise_constant"
    CUBIC_SPLINE = "cubic_spline"
    HERMITE_SPLINE = "hermite_spline"
    LAGRANGE = "lagrange"


class InterpolationDataType(Enum):
    """Enumeration of interpolation data types."""

    SCALAR = "scalar"
    VECTOR = "vector"
    MATRIX = "matrix"


class InterpolationErrorDesigner(ErrorDesignerBase):
    """Interpolation error analysis against benchmark solutions.

    This class analyzes interpolation accuracy by comparing interpolated data against
    high-fidelity benchmark solutions. Supports parameter sweeps for:
    1. Number of data points used for interpolation
    2. Interpolation methods (linear, cubic spline, Lagrange, etc.)
    3. Interpolation orders (for Lagrange interpolation)
    4. Different data types (scalar, vector, matrix)

    """

    def __init__(
        self,
        interpolation_type: InterpolationType = InterpolationType.LAGRANGE,
        data_type: InterpolationDataType = InterpolationDataType.VECTOR,
    ):
        """Initialize the interpolation error designer.

        Args:
            interpolation_type: Type of interpolation to analyze
            data_type: Type of data being interpolated (scalar, vector, matrix)

        Note:
            Always uses ErrorMethod.BENCHMARK for consistent error analysis
        """
        # Force benchmark method for interpolation analysis
        super().__init__(error_method=ErrorMethod.BENCHMARK)

        if not TUDAT_AVAILABLE:
            raise ImportError("TudatPy is required for interpolation analysis")

        self.interpolation_type = interpolation_type
        self.data_type = data_type

        # Cache for interpolation data and benchmark solution
        self._interpolation_data = None
        self._reference_times = None
        self._reference_values = None

        # Timing data storage
        self._timing_data = {}

    def _run_sample(self, parameters: Dict[str, Any]) -> Dict[float, np.ndarray]:
        """Run interpolation analysis for a single parameter set.

        Args:
            parameters: Dictionary containing:
                - 'n_data_points': Number of data points for interpolation
                - 'interpolation_type': InterpolationType enum (optional, overrides default)
                - 'interpolation_order': Order for Lagrange interpolation (optional)

        Returns:
            Dictionary of interpolated state history {time: state_vector}
        """
        n_data_points = parameters["n_data_points"]
        interpolation_type = parameters.get(
            "interpolation_type", self.interpolation_type
        )
        interpolation_order = parameters.get("interpolation_order", None)

        # Ensure benchmark solution exists
        if self._benchmark_solution is None:
            raise ValueError(
                "Benchmark solution must be created before running samples"
            )

        # Get subset of benchmark data for interpolation
        benchmark_times = sorted(self._benchmark_solution.keys())

        # Use _generate_chebyshev_states to get data at exact Chebyshev nodes
        time_start = benchmark_times[0]
        time_end = benchmark_times[-1]

        # Get initial state from benchmark solution
        initial_state = self._benchmark_solution[time_start]

        # Generate states at exact Chebyshev node times
        # Time the data generation process
        data_generation_start_time = time.time()
        interpolation_data = self._generate_chebyshev_states(
            start_epoch=time_start,
            end_epoch=time_end,
            n_points=n_data_points,
            initial_state=initial_state,
        )
        data_generation_end_time = time.time()
        data_generation_duration = data_generation_end_time - data_generation_start_time

        # Split data into translational state, orientation, and angular velocity
        translational_data = {}
        orientation_data = {}
        angular_velocity_data = {}

        for time_key, state in interpolation_data.items():
            # Use helper method to split state vector appropriately
            components = self._split_state_vector(state)
            translational_data[time_key] = components["translational"]
            orientation_data[time_key] = components["orientation"]
            angular_velocity_data[time_key] = components["angular_velocity"]

        # Create separate interpolators for each component
        # This approach allows for different interpolation strategies per component:
        # - Translational: Standard vector interpolation
        # - Orientation: Quaternion interpolation (could use SLERP in future)
        # - Angular velocity: Standard vector interpolation
        current_type = self.interpolation_type
        self.interpolation_type = interpolation_type  # Temporarily override
        try:
            # Create translational interpolator (6D vector)
            self.data_type = InterpolationDataType.VECTOR
            translational_interpolator = self._create_interpolator(
                translational_data, interpolation_order
            )

            # Create orientation interpolator (4D quaternion vector)
            # Note: This uses standard vector interpolation, not SLERP
            # Future enhancement could implement proper quaternion interpolation
            orientation_interpolator = self._create_interpolator(
                orientation_data, interpolation_order
            )

            # Create angular velocity interpolator (3D vector)
            angular_velocity_interpolator = self._create_interpolator(
                angular_velocity_data, interpolation_order
            )
        finally:
            self.interpolation_type = current_type  # Restore original

        # Evaluate all interpolators at benchmark times and combine results
        interpolated_states = {}
        interpolation_start_time = time.time()

        for time_point in benchmark_times:
            try:
                # Interpolate each component separately
                translational_value = translational_interpolator.interpolate(
                    float(time_point)
                )
                orientation_value = orientation_interpolator.interpolate(
                    float(time_point)
                )
                angular_velocity_value = angular_velocity_interpolator.interpolate(
                    float(time_point)
                )

                # Convert to numpy arrays if needed
                if not isinstance(translational_value, np.ndarray):
                    translational_value = np.array(translational_value)
                if not isinstance(orientation_value, np.ndarray):
                    orientation_value = np.array(orientation_value)
                if not isinstance(angular_velocity_value, np.ndarray):
                    angular_velocity_value = np.array(angular_velocity_value)

                # Normalize quaternion to ensure unit length
                orientation_value = self._normalize_quaternion(orientation_value)

                # Combine all components into full state vector
                combined_state = np.concatenate(
                    [translational_value, orientation_value, angular_velocity_value]
                )

                interpolated_states[time_point] = combined_state

            except (ValueError, RuntimeError):
                # Skip times outside interpolation range or other interpolation errors
                continue

        interpolation_end_time = time.time()
        interpolation_duration = interpolation_end_time - interpolation_start_time

        # Store timing information in the result metadata
        self._timing_data[str(parameters)] = {
            "data_generation_time": data_generation_duration,
            "interpolation_time": interpolation_duration,
            "n_evaluations": len(interpolated_states),
            "time_per_evaluation": interpolation_duration
            / max(len(interpolated_states), 1),
            "speedup_factor": data_generation_duration
            / max(interpolation_duration, 1e-9),
            "n_data_points": n_data_points,
            "parameters": parameters.copy(),  # Store original parameters for better formatting
            "evaluations_per_second": len(interpolated_states)
            / max(interpolation_duration, 1e-9),
        }

        return interpolated_states

    def _split_state_vector(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """Split a state vector into translational, orientation, and angular velocity components.

        Args:
            state: Full state vector

        Returns:
            Dictionary with 'translational', 'orientation', 'angular_velocity' components
        """
        if len(state) == 13:
            # Full spacecraft state: [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
            return {
                "translational": state[:6],  # position + velocity
                "orientation": state[6:10],  # quaternion
                "angular_velocity": state[10:13],  # angular velocity
            }
        elif len(state) == 6:
            # Translational only: [x, y, z, vx, vy, vz]
            return {
                "translational": state,
                "orientation": np.array([1.0, 0.0, 0.0, 0.0]),  # identity quaternion
                "angular_velocity": np.array([0.0, 0.0, 0.0]),  # zero angular velocity
            }
        elif len(state) == 7:
            # Position + quaternion: [x, y, z, q0, q1, q2, q3]
            return {
                "translational": np.concatenate(
                    [state[:3], np.zeros(3)]
                ),  # position + zero velocity
                "orientation": state[3:7],
                "angular_velocity": np.array([0.0, 0.0, 0.0]),
            }
        else:
            raise ValueError(
                f"Unsupported state vector length: {len(state)}. Expected 6, 7, or 13 elements."
            )

    def _normalize_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        """Normalize a quaternion to unit length.

        Args:
            quaternion: [q0, q1, q2, q3] quaternion components

        Returns:
            Normalized unit quaternion
        """
        norm = np.linalg.norm(quaternion)
        if norm > 1e-12:  # Avoid division by zero
            return quaternion / norm
        else:
            # Return identity quaternion if input is near zero
            return np.array([1.0, 0.0, 0.0, 0.0])

    def get_timing_summary(self) -> Dict[str, Any]:
        """Get summary of timing data for all parameter combinations.

        Returns:
            Dictionary with timing statistics
        """
        if not self._timing_data:
            return {"message": "No timing data available. Run parameter sweep first."}

        # Aggregate timing statistics
        data_gen_times = [
            data["data_generation_time"] for data in self._timing_data.values()
        ]
        interp_times = [
            data["interpolation_time"] for data in self._timing_data.values()
        ]
        speedup_factors = [
            data["speedup_factor"] for data in self._timing_data.values()
        ]

        return {
            "total_runs": len(self._timing_data),
            "data_generation": {
                "mean_time": np.mean(data_gen_times),
                "std_time": np.std(data_gen_times),
                "total_time": np.sum(data_gen_times),
            },
            "interpolation": {
                "mean_time": np.mean(interp_times),
                "std_time": np.std(interp_times),
                "total_time": np.sum(interp_times),
            },
            "speedup": {
                "mean_factor": np.mean(speedup_factors),
                "max_factor": np.max(speedup_factors),
                "min_factor": np.min(speedup_factors),
            },
            "detailed_data": self._timing_data,
        }

    def print_timing_summary(self):
        """Print formatted timing summary with detailed table."""
        summary = self.get_timing_summary()

        if "message" in summary:
            print(summary["message"])
            return

        print("\n=== INTERPOLATION PERFORMANCE ANALYSIS ===")
        print(f"Total parameter combinations tested: {summary['total_runs']}")

        # Detailed table for each interpolator configuration
        print("\nDETAILED PERFORMANCE TABLE:")
        print("-" * 110)
        print(
            f"{'Configuration':<30} {'Data Gen (s)':<12} {'Interp (s)':<12} {'Speedup':<10} {'Eval/s':<12} {'Data Pts':<10}"
        )
        print("-" * 110)

        # Sort by data points for better readability
        sorted_items = sorted(
            summary["detailed_data"].items(), key=lambda x: x[1].get("n_data_points", 0)
        )

        for param_str, data in sorted_items:
            # Use stored parameters for better display
            if "parameters" in data:
                param_display = self._format_parameters_dict(data["parameters"])
            else:
                param_display = self._format_parameter_string(param_str)

            data_gen_time = data["data_generation_time"]
            interp_time = data["interpolation_time"]
            speedup = data["speedup_factor"]
            evals_per_sec = data.get(
                "evaluations_per_second", data["n_evaluations"] / max(interp_time, 1e-9)
            )
            n_data_points = data["n_data_points"]

            print(
                f"{param_display:<30} {data_gen_time:<12.4f} {interp_time:<12.6f} "
                f"{speedup:<10.1f} {evals_per_sec:<12.0f} {n_data_points:<10}"
            )

        print("-" * 110)

        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print("\nDATA GENERATION:")
        print(
            f"  Mean time: {summary['data_generation']['mean_time']:.4f} ± {summary['data_generation']['std_time']:.4f} s"
        )
        print(f"  Total time: {summary['data_generation']['total_time']:.4f} s")

        print("\nINTERPOLATION EVALUATION:")
        print(
            f"  Mean time: {summary['interpolation']['mean_time']:.6f} ± {summary['interpolation']['std_time']:.6f} s"
        )
        print(f"  Total time: {summary['interpolation']['total_time']:.6f} s")

        print("\nSPEEDUP ANALYSIS:")
        print(f"  Mean speedup factor: {summary['speedup']['mean_factor']:.1f}x")
        print(f"  Max speedup factor: {summary['speedup']['max_factor']:.1f}x")
        print(f"  Min speedup factor: {summary['speedup']['min_factor']:.1f}x")

        print("\nINTERPRETATION:")
        print(
            f"  On average, interpolation is {summary['speedup']['mean_factor']:.1f}x faster"
        )
        print("  than generating the same data points from scratch.")
        print("  Higher speedup factors indicate better computational efficiency.")
        print("=" * 50)

    def _format_parameter_string(self, param_str: str) -> str:
        """Format parameter string for display in table.

        Args:
            param_str: String representation of parameters dictionary

        Returns:
            Formatted string for table display
        """
        try:
            # Try to extract key parameters from the string representation
            if "'n_data_points'" in param_str:
                # Extract n_data_points value
                import re

                n_points_match = re.search(r"'n_data_points': (\d+)", param_str)
                n_points = n_points_match.group(1) if n_points_match else "?"

                # Extract interpolation type if present
                type_match = re.search(
                    r"'interpolation_type': InterpolationType\.(\w+)", param_str
                )
                interp_type = type_match.group(1) if type_match else ""

                # Extract interpolation order if present
                order_match = re.search(r"'interpolation_order': (\d+)", param_str)
                order = f"_ord{order_match.group(1)}" if order_match else ""

                if interp_type:
                    return f"{interp_type}{order}_n{n_points}"
                else:
                    return f"n_points_{n_points}"
            else:
                # Fallback to truncated string
                return param_str[:23] + ".." if len(param_str) > 25 else param_str
        except (ValueError, KeyError, AttributeError):
            # Safe fallback for specific exceptions
            return param_str[:23] + ".." if len(param_str) > 25 else param_str

    def _format_parameters_dict(self, params: Dict[str, Any]) -> str:
        """Format parameters dictionary for display in table.

        Args:
            params: Parameters dictionary

        Returns:
            Formatted string for table display
        """
        parts = []

        # Add interpolation type if present
        if "interpolation_type" in params:
            interp_type = params["interpolation_type"]
            if hasattr(interp_type, "value"):
                parts.append(interp_type.value.upper())
            else:
                parts.append(str(interp_type).upper())

        # Add interpolation order if present
        if "interpolation_order" in params:
            parts.append(f"ord={params['interpolation_order']}")

        # Add number of data points
        if "n_data_points" in params:
            parts.append(f"n={params['n_data_points']}")

        # Add any other parameters (for multivariate sweeps)
        for key, value in params.items():
            if key not in [
                "interpolation_type",
                "interpolation_order",
                "n_data_points",
            ]:
                if hasattr(value, "value"):  # Enum
                    parts.append(f"{key}={value.value}")
                else:
                    parts.append(f"{key}={value}")

        result = "_".join(parts) if parts else "default"

        # Truncate if too long
        return result[:28] + ".." if len(result) > 30 else result

    def clear_timing_data(self):
        """Clear stored timing data."""
        self._timing_data = {}

    def export_timing_data_csv(self, filename: str = None):
        """Export timing data to CSV file.

        Args:
            filename: Output filename (auto-generated if None)
        """
        if not self._timing_data:
            print("No timing data available to export.")
            return

        if filename is None:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"interpolation_timing_analysis_{timestamp}.csv"

        import csv

        # Prepare data for CSV
        fieldnames = [
            "configuration",
            "interpolation_type",
            "interpolation_order",
            "n_data_points",
            "data_generation_time",
            "interpolation_time",
            "n_evaluations",
            "time_per_evaluation",
            "evaluations_per_second",
            "speedup_factor",
        ]

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for _, data in self._timing_data.items():
                row = {
                    "configuration": self._format_parameters_dict(
                        data.get("parameters", {})
                    ),
                    "data_generation_time": data["data_generation_time"],
                    "interpolation_time": data["interpolation_time"],
                    "n_evaluations": data["n_evaluations"],
                    "time_per_evaluation": data["time_per_evaluation"],
                    "evaluations_per_second": data.get("evaluations_per_second", 0),
                    "speedup_factor": data["speedup_factor"],
                    "n_data_points": data["n_data_points"],
                }

                # Extract individual parameter values
                if "parameters" in data:
                    params = data["parameters"]
                    if "interpolation_type" in params:
                        interp_type = params["interpolation_type"]
                        row["interpolation_type"] = (
                            interp_type.value
                            if hasattr(interp_type, "value")
                            else str(interp_type)
                        )
                    else:
                        row["interpolation_type"] = "default"

                    row["interpolation_order"] = params.get(
                        "interpolation_order", "N/A"
                    )
                else:
                    row["interpolation_type"] = "unknown"
                    row["interpolation_order"] = "unknown"

                writer.writerow(row)

        print(f"Timing data exported to: {filename}")
        return filename

    def _generate_chebyshev_states(
        self, start_epoch, end_epoch, n_points: int, initial_state: np.ndarray
    ) -> Dict[float, np.ndarray]:
        """Compute states at Chebyshev points by propagating between nodes.

        Args:
            start_epoch: Start time for the trajectory
            end_epoch: End time for the trajectory
            n_points: Number of Chebyshev points to generate
            initial_state: Initial state vector at start_epoch

        Returns:
            Dictionary with time as key and state as value:
            {time_point: state_vector, ...}
        """
        chebyshev_points = chebpts2(n_points)
        # Scale points to the epoch range
        time_points = (
            0.5 * (end_epoch - start_epoch) * (chebyshev_points + 1) + start_epoch
        )

        # Initialize states dictionary
        chebyshev_states = {}
        chebyshev_states[time_points[0]] = (
            initial_state  # First state is the initial condition
        )

        # Propagate between consecutive Chebyshev points
        for i in range(n_points - 1):
            # Parameters for propagation from time_points[i] to time_points[i+1]
            propagation_params = {
                "start_epoch": time_points[i],
                "end_epoch": time_points[i + 1],
                "initial_state": chebyshev_states[time_points[i]],
            }

            # Run simulation to get state at next Chebyshev point
            # Use a separate propagation method to avoid circular dependency with _run_sample
            sample_result = self._propagate_trajectory(propagation_params)

            # Extract final state from the propagation result
            # The _propagate_trajectory method should return a dict with 'state_history'
            state_history = sample_result["state_history"]
            final_time = max(state_history.keys())
            chebyshev_states[time_points[i + 1]] = state_history[final_time]

        return chebyshev_states

    @abstractmethod
    def _propagate_trajectory(
        self, parameters: Dict[str, Any]
    ) -> Dict[str, Dict[float, np.ndarray]]:
        """Propagate trajectory between two time points.

        This method must be implemented by subclasses to provide trajectory propagation
        for generating Chebyshev node data.

        Args:
            parameters: Dictionary containing:
                - 'start_epoch': Start time for propagation
                - 'end_epoch': End time for propagation
                - 'initial_state': Initial state vector at start_epoch

        Returns:
            Dictionary containing 'state_history': {time: state_vector}
        """
        raise NotImplementedError(
            "Subclasses must implement _propagate_trajectory for Chebyshev node generation"
        )

    def _create_interpolator(
        self,
        data: Dict[float, Union[float, np.ndarray]],
        interpolation_order: Optional[int] = None,
    ) -> Any:
        """Create a TudatPy interpolator based on configuration.

        Args:
            data: Dictionary mapping time to value {time: numpy array}
            interpolation_order: Order for Lagrange interpolation (if applicable)

        Returns:
            TudatPy interpolator object
        """
        # Handle different interpolation types
        if self.interpolation_type == InterpolationType.LINEAR:
            interpolation_settings = interpolators.linear_interpolation()
        elif self.interpolation_type == InterpolationType.PIECEWISE_CONSTANT:
            interpolation_settings = interpolators.piecewise_constant_interpolation()
        elif self.interpolation_type == InterpolationType.CUBIC_SPLINE:
            interpolation_settings = interpolators.cubic_spline_interpolation()
        elif self.interpolation_type == InterpolationType.HERMITE_SPLINE:
            interpolation_settings = interpolators.hermite_spline_interpolation()
        elif self.interpolation_type == InterpolationType.LAGRANGE:
            if interpolation_order is None:
                # Ensure interpolation_order is even and <= len(data) - 1, max 8
                max_order = min(len(data) - 1, 8)
                interpolation_order = max_order - (max_order % 2)
            interpolation_settings = interpolators.lagrange_interpolation(
                interpolation_order
            )
        else:
            raise ValueError(f"Unknown interpolation type: {self.interpolation_type}")

        # Ensure all time keys are float type for TudatPy
        formatted_data = {float(time_key): value for time_key, value in data.items()}

        # Create interpolator based on data type
        if self.data_type == InterpolationDataType.SCALAR:
            return interpolators.create_one_dimensional_scalar_interpolator(
                formatted_data, interpolation_settings
            )

        elif self.data_type == InterpolationDataType.VECTOR:
            return interpolators.create_one_dimensional_vector_interpolator(
                formatted_data, interpolation_settings
            )

        elif self.data_type == InterpolationDataType.MATRIX:
            return interpolators.create_one_dimensional_matrix_interpolator(
                formatted_data, interpolation_settings
            )
        else:
            raise ValueError(f"Unknown data type: {self.data_type}")
