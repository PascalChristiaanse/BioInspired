"""Error designer base class.
This class serves as a base for error designer implementations that perform parameter sweeps
and error analysis across different configurations (timesteps, tolerances, interpolation orders, etc.).
"""

import numpy as np
import time
from enum import Enum
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from dataclasses import dataclass


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


@dataclass
class ParameterSweepConfig:
    """Configuration for parameter sweeps."""

    parameter_name: str
    parameter_values: Union[List, np.ndarray]
    parameter_display_name: Optional[str] = None
    parameter_units: Optional[str] = None


class SweepMode(Enum):
    """Enumeration of parameter sweep modes."""

    CARTESIAN = "cartesian"
    PAIRED = "paired"
    LATIN_HYPERCUBE = "latin_hypercube"


@dataclass
class SweepConfig:
    """Configuration for parameter sweeps with n variables."""

    parameters: List[ParameterSweepConfig]  # All parameters
    sweep_mode: SweepMode = SweepMode.CARTESIAN  # Enum for sweep mode

    @property
    def n_dimensions(self) -> int:
        """Get the number of parameter dimensions."""
        return len(self.parameters)

    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        if self.sweep_mode == SweepMode.CARTESIAN:
            total = 1
            for param in self.parameters:
                total *= len(param.parameter_values)
            return total
        elif self.sweep_mode == SweepMode.PAIRED:
            # All parameters must have same length
            return len(self.parameters[0].parameter_values)
        elif self.sweep_mode == SweepMode.LATIN_HYPERCUBE:
            # Number of samples equals minimum parameter length
            return min(len(p.parameter_values) for p in self.parameters)
        else:
            raise ValueError(f"Unknown sweep mode: {self.sweep_mode}")


@dataclass
class ErrorAnalysisResult:
    """Results from error analysis."""

    parameter_values: np.ndarray
    error_metrics: Dict[
        str, Dict[str, Dict[float, float]]
    ]  # e.g., {"position_error": {combo_key: {time: error}, ...}}
    computation_times: np.ndarray
    metadata: Dict[str, Any]
    parameter_combinations: List[Dict[str, Any]]  # Required - parameter combinations


class ErrorMethod(Enum):
    """Enumeration of error computation methods."""

    ANALYTICAL = "analytical"
    RICHARDSON = "richardson"
    BENCHMARK = "benchmark"


class ErrorDesignerBase(ABC):
    """Base class for error designers that perform parameter sweeps and error analysis.

    This class provides a framework for:
    1. Parameter sweeping (timesteps, tolerances, interpolation orders, etc.)
    2. Error computation using different methods (Richardson, benchmark, analytical)
    3. Results visualization and analysis
    4. Convergence analysis

    Subclasses must implement specific simulation/computation logic.
    """

    def __init__(self, error_method: ErrorMethod = ErrorMethod.RICHARDSON):
        """Initialize the error designer.

        Args:
            error_method: Method for error computation
        """
        self.error_method = error_method
        self._benchmark_solution = None
        self._analytical_solution = None
        self._results_cache = {}

    @abstractmethod
    def _run_sample(self, parameters: Dict[str, Any]) -> Dict[float, np.ndarray]:
        """Run a single sample with given parameters.

        Args:
            parameters (Dict[str, Any]): Dictionary of parameter values for this sample

        Returns:
            state_history (Dict[np.ndarray]): Dictionary containing results
        """
        raise NotImplementedError("Subclasses must implement the _run_sample method.")

    def _create_benchmark_solution(self) -> Dict[float, np.ndarray]:
        """Create high-fidelity benchmark solution.

        This method must be implemented by subclasses to provide the reference solution
        against which interpolation accuracy will be measured.

        Returns:
            Dictionary of benchmark state history {time: state_vector}
        """
        raise NotImplementedError(
            "Subclasses must implement _create_benchmark_solution to provide reference data"
        )

    def _create_analytical_solution(self) -> Dict[float, np.ndarray]:
        """Create an analytical solution if available.

        Returns:
            state_history (Dict[np.ndarray]): Dictionary containing analytical results
        """
        raise NotImplementedError(
            "Subclasses must implement the _create_analytical_solution method."
        )

    def perform_parameter_sweep(
        self,
        sweep_config: SweepConfig,
        fixed_parameters: Optional[Dict[str, Any]] = None,
    ) -> ErrorAnalysisResult:
        """Perform a parameter sweep with n dimensions.

        Args:
            sweep_config: Configuration for the parameter sweep
            fixed_parameters: Fixed parameters for all samples

        Returns:
            ErrorAnalysisResult containing the analysis results
        """
        if fixed_parameters is None:
            fixed_parameters = {}

        # Generate parameter combinations
        parameter_combinations = self._generate_parameter_combinations(sweep_config)
        n_samples = len(parameter_combinations)

        print("=" * 80)
        print(f"PARAMETER SWEEP ANALYSIS ({sweep_config.n_dimensions}D)")
        print("=" * 80)
        print(
            f"Parameters: {[p.parameter_display_name or p.parameter_name for p in sweep_config.parameters]}"
        )
        print(f"Dimensions: {sweep_config.n_dimensions}")
        print(f"Sweep mode: {sweep_config.sweep_mode}")
        print(f"Total combinations: {n_samples}")
        print(f"Error method: {self.error_method.value.upper()}")

        # Show parameter space size
        param_sizes = [len(p.parameter_values) for p in sweep_config.parameters]
        print(
            f"Parameter space: {' × '.join(map(str, param_sizes))} = {sweep_config.total_combinations()}"
        )

        # Prepare benchmark/analytical solution if needed
        self._prepare_reference_solution()

        # Initialize results storage
        error_metrics = {}
        computation_times = np.zeros(n_samples)

        print(f"\nRunning {n_samples} parameter combinations...")
        print("-" * 60)

        # Run parameter sweep
        for i, param_combination in enumerate(parameter_combinations):
            # Prepare parameters for this sample
            sample_parameters = fixed_parameters.copy()
            sample_parameters.update(param_combination)

            # Create display string for this combination
            param_display = self._format_parameter_combination(param_combination)
            print(f"Sample {i + 1}/{n_samples}: {param_display}")

            # Run the sample
            start_time = time.perf_counter()
            sample_result = self._run_sample(sample_parameters)
            computation_times[i] = time.perf_counter() - start_time

            # Compute errors
            errors = self._compute_errors(sample_result, sample_parameters)

            # Store error metrics using combination index
            combination_key = f"combo_{i}"
            for error_type, error_time_series in errors.items():
                if error_type not in error_metrics:
                    error_metrics[error_type] = {}
                error_metrics[error_type][combination_key] = error_time_series

            print(f"  Runtime: {computation_times[i]:.4f}s")

        print("\n" + "=" * 60)
        print(f"{sweep_config.n_dimensions}D PARAMETER SWEEP COMPLETE")

        # Create result object - use first parameter for primary values
        primary_param_name = sweep_config.parameters[0].parameter_name
        primary_values = [combo[primary_param_name] for combo in parameter_combinations]

        result = ErrorAnalysisResult(
            parameter_values=np.array(primary_values),
            error_metrics=error_metrics,
            computation_times=computation_times,
            parameter_combinations=parameter_combinations,
            metadata={
                "parameter_name": sweep_config.parameters[0].parameter_name,
                "parameter_display_name": sweep_config.parameters[
                    0
                ].parameter_display_name,
                "parameter_units": sweep_config.parameters[0].parameter_units,
                "all_parameters": [
                    {
                        "name": p.parameter_name,
                        "display_name": p.parameter_display_name,
                        "units": p.parameter_units,
                        "values": list(p.parameter_values),
                    }
                    for p in sweep_config.parameters
                ],
                "n_dimensions": sweep_config.n_dimensions,
                "sweep_mode": sweep_config.sweep_mode.value,
                "parameter_space_size": param_sizes,
                "total_combinations": sweep_config.total_combinations(),
                "error_method": self.error_method.value,
                "fixed_parameters": fixed_parameters,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return result

    def _prepare_reference_solution(self):
        """Prepare benchmark or analytical solution if needed."""
        if (
            self.error_method == ErrorMethod.BENCHMARK
            and self._benchmark_solution is None
        ):
            print("\nCreating benchmark solution...")
            self._benchmark_solution = self._create_benchmark_solution()

        elif (
            self.error_method == ErrorMethod.ANALYTICAL
            and self._analytical_solution is None
        ):
            print("\nCreating analytical solution...")
            analytical_solution = self._create_analytical_solution()
            if analytical_solution is None:
                raise ValueError("Analytical solution not available for this problem")
            self._analytical_solution = analytical_solution

    def _generate_parameter_combinations(
        self, sweep_config: SweepConfig
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for multivariate sweep.

        Args:
            sweep_config: Multivariate sweep configuration

        Returns:
            List of parameter dictionaries
        """
        if sweep_config.sweep_mode == SweepMode.CARTESIAN:
            return self._generate_cartesian_combinations(sweep_config)
        elif sweep_config.sweep_mode == SweepMode.PAIRED:
            return self._generate_paired_combinations(sweep_config)
        elif sweep_config.sweep_mode == SweepMode.LATIN_HYPERCUBE:
            return self._generate_latin_hypercube_combinations(sweep_config)
        else:
            raise ValueError(f"Unknown sweep mode: {sweep_config.sweep_mode}")

    def _generate_cartesian_combinations(
        self, sweep_config: SweepConfig
    ) -> List[Dict[str, Any]]:
        """Generate Cartesian product of parameter values for n dimensions.

        Args:
            sweep_config: Multivariate sweep configuration

        Returns:
            List of parameter dictionaries
        """
        import itertools

        # Use the new parameters list for n-dimensional support
        param_names = [p.parameter_name for p in sweep_config.parameters]
        param_values = [list(p.parameter_values) for p in sweep_config.parameters]

        combinations = []
        for value_combo in itertools.product(*param_values):
            combo_dict = dict(zip(param_names, value_combo))
            combinations.append(combo_dict)

        return combinations

    def _generate_paired_combinations(
        self, sweep_config: SweepConfig
    ) -> List[Dict[str, Any]]:
        """Generate paired parameter combinations for n dimensions (same index across all parameters).

        Args:
            sweep_config: Multivariate sweep configuration

        Returns:
            List of parameter dictionaries
        """
        # Use the new parameters list for n-dimensional support
        all_params = sweep_config.parameters

        # Ensure all parameter lists have the same length
        param_lengths = [len(p.parameter_values) for p in all_params]
        if len(set(param_lengths)) > 1:
            raise ValueError(
                f"For paired mode, all parameter lists must have the same length. Got lengths: {param_lengths}"
            )

        n_combinations = param_lengths[0]

        combinations = []
        for i in range(n_combinations):
            combo_dict = {}
            for param in all_params:
                combo_dict[param.parameter_name] = param.parameter_values[i]
            combinations.append(combo_dict)

        return combinations

    def _generate_latin_hypercube_combinations(
        self, sweep_config: SweepConfig
    ) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube sampling combinations for n dimensions.

        Args:
            sweep_config: Multivariate sweep configuration

        Returns:
            List of Latin Hypercube sampled parameter combinations
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("scipy is required for Latin Hypercube sampling")

        # Use minimum parameter length as sample size
        n_samples = min(len(p.parameter_values) for p in sweep_config.parameters)
        n_dimensions = sweep_config.n_dimensions

        # Generate Latin Hypercube samples
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=42)
        samples = sampler.random(n=n_samples)

        combinations = []
        for sample in samples:
            param_dict = {}
            for i, param in enumerate(sweep_config.parameters):
                # Map sample value [0,1] to parameter range
                param_min = min(param.parameter_values)
                param_max = max(param.parameter_values)
                scaled_value = param_min + sample[i] * (param_max - param_min)
                param_dict[param.parameter_name] = scaled_value
            combinations.append(param_dict)

        return combinations

    def _format_parameter_combination(self, param_combination: Dict[str, Any]) -> str:
        """Format parameter combination for display.

        Args:
            param_combination: Dictionary of parameter values

        Returns:
            Formatted string
        """
        param_strings = []
        for key, value in param_combination.items():
            if hasattr(value, "value"):  # Enum
                param_strings.append(f"{key}={value.value}")
            else:
                param_strings.append(f"{key}={value}")

        return ", ".join(param_strings)

    def _compute_errors(
        self, sample_result: Dict[float, np.ndarray], parameters: Dict[str, Any] = None
    ) -> Dict[str, Dict[float, float]]:
        """Compute errors for a sample result based on the error method.

        Args:
            sample_result: Result from _run_sample
            parameters: Parameters used for this sample

        Returns:
            Dictionary of time-indexed error histories
        """
        if self.error_method == ErrorMethod.RICHARDSON:
            if parameters is None:
                raise ValueError(
                    "Parameters must be provided for Richardson extrapolation"
                )
            return self._compute_richardson_errors(sample_result, parameters)
        elif self.error_method == ErrorMethod.BENCHMARK:
            return self._compute_benchmark_errors(sample_result)
        elif self.error_method == ErrorMethod.ANALYTICAL:
            return self._compute_analytical_errors(sample_result)
        else:
            raise ValueError(f"Unknown error method: {self.error_method}")

    def _compute_richardson_errors(
        self, sample_result: Dict[float, np.ndarray], parameters: Dict[str, Any]
    ) -> Dict[str, Dict[float, float]]:
        """Compute errors using Richardson extrapolation (half-parameter comparison).

        Args:
            sample_result: State history from _run_sample
            parameters: Parameters used for this sample

        Returns:
            Dictionary of time-indexed error histories
        """
        # Create parameters with half the main parameter value
        # This is problem-specific, so subclasses may need to override
        half_parameters = parameters.copy()

        # Find the swept parameter and halve it
        for key, value in parameters.items():
            if isinstance(value, (int, float)) and value > 0:
                # Assume this is the parameter being swept
                half_parameters[key] = value / 2.0
                break

        # Run with half parameter
        half_result = self._run_sample(half_parameters)

        # Compute error time series between complete state histories
        return self._compute_state_error(sample_result, half_result)

    def _compute_benchmark_errors(
        self, sample_result: Dict[float, np.ndarray]
    ) -> Dict[str, Dict[float, float]]:
        """Compute errors against benchmark solution.

        Args:
            sample_result: State history from _run_sample

        Returns:
            Dictionary of time-indexed error histories
        """
        if self._benchmark_solution is None:
            raise ValueError("Benchmark solution not available")

        # Compute error time series between complete state histories
        return self._compute_state_error(sample_result, self._benchmark_solution)

    def _compute_analytical_errors(
        self, sample_result: Dict[float, np.ndarray]
    ) -> Dict[str, Dict[float, float]]:
        """Compute errors against analytical solution.

        Args:
            sample_result: State history from _run_sample

        Returns:
            Dictionary of time-indexed error histories
        """
        if self._analytical_solution is None:
            raise ValueError("Analytical solution not available")

        # Compute error time series between complete state histories
        return self._compute_state_error(sample_result, self._analytical_solution)

    def _compute_state_error(
        self,
        state_history1: Dict[float, np.ndarray],
        state_history2: Dict[float, np.ndarray],
    ) -> Dict[str, Dict[float, float]]:
        """Compute the error between two state histories over their entire trajectories.

        Args:
            state_history1: First state history {time: state_vector}
            state_history2: Second state history {time: state_vector}

        Returns:
            Dictionary of time-indexed error histories {error_type: {time: error_value}}
        """
        # Find common time points for comparison
        times1 = set(state_history1.keys())
        times2 = set(state_history2.keys())
        common_times = sorted(times1.intersection(times2))

        if not common_times:
            raise ValueError("No common time points found between state histories")

        # Initialize error time series storage
        error_time_series = {
            "position_error": {},
            "velocity_error": {},
            "quaternion_error": {},
            "angular_velocity_error": {},
            "angular_rate_error": {},
            "orientation_error": {},
        }

        for time_point in common_times:
            state1 = state_history1[time_point]
            state2 = state_history2[time_point]

            # Handle different state vector lengths
            min_len = min(len(state1), len(state2))

            # Position error (first 3 components)
            if min_len >= 3:
                pos_error = np.linalg.norm(state1[:3] - state2[:3])
                error_time_series["position_error"][time_point] = pos_error

            # Velocity error (next 3 components)
            if min_len >= 6:
                vel_error = np.linalg.norm(state1[3:6] - state2[3:6])
                error_time_series["velocity_error"][time_point] = vel_error

            # Quaternion error (next 4 components)
            if min_len >= 10:
                quat_error = quaternion_angular_error(state1[6:10], state2[6:10])
                error_time_series["quaternion_error"][time_point] = quat_error

            # Angular velocity error (last 3 components)
            if min_len >= 13:
                ang_vel_error = np.linalg.norm(state1[10:13] - state2[10:13])
                error_time_series["angular_velocity_error"][time_point] = ang_vel_error

                # Angular rate error (magnitude of angular velocity vectors)
                state1_angular_rate = np.linalg.norm(state1[10:13])
                state2_angular_rate = np.linalg.norm(state2[10:13])
                ang_rate_error = abs(state1_angular_rate - state2_angular_rate)
                error_time_series["angular_rate_error"][time_point] = ang_rate_error

        # Remove empty error time series
        error_time_series = {k: v for k, v in error_time_series.items() if v}

        return error_time_series
