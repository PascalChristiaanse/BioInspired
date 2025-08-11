"""Stop Neuron Basic Problem module.
This extends the basic problem with stop neuron capability:
 - In plane (XY+R_Z) problem
 - Two thrusters + one stop neuron
 - Empty universe
 - No rotation on target
 - 6 input neurons based on J. Leitner (2010)
 - 10 hidden neurons
 - 2 + 1 output neurons (with stop neuron)
"""

import numpy as np
import numba as nb
from functools import lru_cache
from overrides import override

from tudatpy.numerical_simulation.propagation import (
    create_dependent_variable_dictionary,
)
from tudatpy.numerical_simulation.propagation_setup import dependent_variable
from tudatpy.astro.element_conversion import (
    rotation_matrix_to_quaternion_entries,
    quaternion_entries_to_rotation_matrix,
)

from bioinspired.problem import ProblemBase
from bioinspired.problem import JLeitner2010

from bioinspired.spacecraft import Lander2, Endurance, EphemerisSpacecraft
from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.controller import MLPController

import logging

# Suppress numba logging messages
logging.getLogger("numba").setLevel(logging.WARNING)


@nb.jit(nopython=True, cache=True)
def rotation_matrix_to_euler_angles(rotation_matrix):
    """Convert a rotation matrix to Euler angles (roll, pitch, yaw)."""
    sy = np.sqrt(
        rotation_matrix[0, 0] * rotation_matrix[0, 0]
        + rotation_matrix[1, 0] * rotation_matrix[1, 0]
    )
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = np.arctan2(-rotation_matrix[2, 0], sy)
        z = 0

    return np.array([x, y, z])


@nb.jit(nopython=True, cache=True)
def compute_state_vector_jit(
    lander_state,
    target_state,
    lander_orientation,
    target_orientation,
    lander_angular_rate_matrix,
    target_angular_rate_matrix,
):
    """JIT-compiled computation of state vector differences.

    Args:
        lander_state: [x, y, z, vx, vy, vz] of lander
        target_state: [x, y, z, vx, vy, vz] of target
        lander_orientation: 3x3 rotation matrix
        target_orientation: 3x3 rotation matrix
        lander_angular_rate_matrix: 3x3 angular rate matrix
        target_angular_rate_matrix: 3x3 angular rate matrix

    Returns:
        6-element state vector [x_diff, y_diff, x_dot_diff, y_dot_diff, z_angle_diff, z_rate_diff]
    """
    # Position and velocity differences
    x_diff = target_state[0] - lander_state[0]
    y_diff = target_state[1] - lander_state[1]
    x_dot_diff = target_state[3] - lander_state[3]
    y_dot_diff = target_state[4] - lander_state[4]

    # Angular differences - extract Z angles from rotation matrices
    lander_z_angle = rotation_matrix_to_euler_angles(lander_orientation)[2]
    target_z_angle = rotation_matrix_to_euler_angles(target_orientation)[2]
    z_angle_diff = target_z_angle - lander_z_angle

    # Angular rate differences - extract Z rates
    lander_angular_rate = rotation_matrix_to_euler_angles(lander_angular_rate_matrix)
    target_angular_rate = rotation_matrix_to_euler_angles(target_angular_rate_matrix)
    z_rate_diff = target_angular_rate[2] - lander_angular_rate[2]

    return np.array([x_diff, y_diff, x_dot_diff, y_dot_diff, z_angle_diff, z_rate_diff])


@nb.jit(nopython=True, cache=True)
def apply_control_vector_jit(control_vector):
    """JIT-compiled control vector application to thrust vector.

    Args:
        control_vector: 2-element control output from neural network

    Returns:
        24-element thrust vector with control applied to thrusters 0, 4, 9, and 12
    """
    modified_thrust_vector = np.zeros(24)
    modified_thrust_vector[0] = control_vector[0]
    modified_thrust_vector[12] = control_vector[0]
    modified_thrust_vector[9] = control_vector[1]
    modified_thrust_vector[4] = control_vector[1]
    return modified_thrust_vector


class StopNeuronMixin:
    """Mixin to add stop neuron capability to any controller."""

    def __init__(
        self,
        simulator,
        stop_threshold: float = 0.5,
        **kwargs,
    ):
        self.stop_threshold = stop_threshold
        simulator.add_termination_condition(
            {
                "type": "propagator.PropagationCustomTerminationSettings",
                "condition": self.should_stop_simulation,
            }
        )

    def should_stop_simulation(self, current_time):
        """Check if simulation should stop based on stop neuron output."""
        # Get the full control vector including stop neuron
        state_vector = self.extract_state_vector(current_time)
        control_vector = self.forward(state_vector)
        control_vector = control_vector.detach().numpy()

        # Stop neuron is the last output
        stop_signal = control_vector[2] if len(control_vector) > 2 else 0
        # if stop_signal > self.stop_threshold:
            # print(f"Stopping simulation at time {current_time} with stop signal: {stop_signal}")
        return stop_signal > self.stop_threshold

    def get_thrust_only(self, current_time):
        """Get just the thrust commands without stop neuron."""
        state_vector = self.extract_state_vector(current_time)
        control_vector = self.forward(state_vector)
        control_vector = control_vector.detach().numpy()

        # Return only the first 2 elements (thrust commands)
        return apply_control_vector_jit(control_vector[:2])


class StopNeuronBasicController(MLPController, StopNeuronMixin):
    """Basic controller with stop neuron for the automatic rendezvous and docking (AR&D) problem."""

    def __init__(
        self,
        simulator,
        lander_name,
        target_name,
        stop_threshold=0.5,
        **kwargs,
    ):
        """Initialize the basic controller with stop neuron."""
        super().__init__(
            input_size=6,  # input_size
            hidden_sizes=[10],  # hidden_sizes
            output_size=3,  # output_size (2 thrusters + 1 stop neuron)
            simulator=simulator,
            lander_name=lander_name,
            target_name=target_name,
            stop_threshold=stop_threshold,
            **kwargs,
        )
        StopNeuronMixin.__init__(self, simulator, stop_threshold=stop_threshold, **kwargs)

    @override
    def extract_state_vector(self, current_time) -> np.ndarray:
        """Extract state vector with JIT optimization for computational parts."""
        # Fetch data from TudatPy (can't be JIT compiled)
        lander_body = self.simulator.get_body_model().get(self.lander_name)
        lander_state = (
            lander_body.state if hasattr(lander_body, "state") else np.zeros(6)
        )

        target_body = self.simulator.get_body_model().get(self.target_name)
        target_state = target_body.ephemeris.cartesian_state(current_time)

        lander_orientation = getattr(
            lander_body, "body_fixed_to_inertial_frame", np.eye(3)
        )
        # Ensure we have a proper 3x3 matrix
        if len(lander_orientation) == 9:
            lander_orientation = lander_orientation.reshape(3, 3)
        elif lander_orientation.shape != (3, 3):
            lander_orientation = np.eye(3)

        target_orientation = target_body.rotation_model.body_fixed_to_inertial_rotation(
            current_time
        )

        lander_angular_rate_matrix = getattr(
            lander_body, "inertial_to_body_fixed_frame", np.eye(3)
        )
        # Ensure we have a proper 3x3 matrix
        if len(lander_angular_rate_matrix) == 9:
            lander_angular_rate_matrix = lander_angular_rate_matrix.reshape(3, 3)
        elif lander_angular_rate_matrix.shape != (3, 3):
            lander_angular_rate_matrix = np.eye(3)

        target_angular_rate_matrix = (
            target_body.rotation_model.time_derivative_body_fixed_to_inertial_rotation(
                current_time
            )
        )

        # Use JIT-compiled computation for the heavy mathematical work
        return compute_state_vector_jit(
            lander_state[:6],  # Ensure we have exactly 6 elements
            target_state[:6],
            lander_orientation,
            target_orientation,
            lander_angular_rate_matrix,
            target_angular_rate_matrix,
        )

    @lru_cache(maxsize=8)
    @override
    def get_control_action(self, current_time):
        """Return the thrust vector (excluding stop neuron) with JIT optimization."""
        return self.get_thrust_only(current_time)


class StopNeuronBasicProblem(ProblemBase):
    """Basic problem with stop neuron capability for the automatic rendezvous and docking (AR&D) problem."""

    def __init__(self, stop_threshold=0.5, max_simulation_time=50):
        super().__init__(JLeitner2010())
        self.stop_threshold = stop_threshold
        self.max_simulation_time = max_simulation_time
        # Make the problem stateless - no instance variables to store objects

    def _create_simulator_components(self):
        """Create fresh simulator components for each evaluation."""
        simulator = EmptyUniverseSimulator(
            dependent_variables_list=[
                dependent_variable.relative_distance("Lander 2", "Endurance-Ephemeris"),
                dependent_variable.relative_speed("Lander 2", "Endurance-Ephemeris"),
            ],
            initial_timestep=0.1,
        )

        endurance = EphemerisSpacecraft(
            simulator,
            Endurance,
            np.array([25, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
        )

        simulator.add_dependent_variable(
            dependent_variable.custom_dependent_variable(endurance.get_orientation, 9)
        )

        endurance.load_ephemeris("./endurance_ephemeris.dat")

        translational_state = np.array([0, 0, 0, 0, 0, 0])  # Initial position (x, y, z)
        orientation = np.eye(3)
        angular_velocity = np.array([0.0, 0, 0])
        orientation_matrix = rotation_matrix_to_quaternion_entries(orientation)

        initial_state = np.concatenate(
            (translational_state, orientation_matrix, angular_velocity)
        )

        controller = StopNeuronBasicController(
            simulator=simulator,
            lander_name="Lander 2",
            target_name="Endurance-Ephemeris",
            stop_threshold=self.stop_threshold,
            max_simulation_time=self.max_simulation_time,
        )

        spacecraft = Lander2(
            initial_state=initial_state,
            simulation=simulator,
            controller=controller,
        )

        return simulator, endurance, controller, spacecraft

    def fitness(self, design):
        """Evaluate the fitness of a solution."""
        # Create fresh components for each evaluation
        simulator, endurance, controller, spacecraft = (
            self._create_simulator_components()
        )

        if design is not None:
            controller.set_weights(design)

        dynamics_simulator = simulator.run(0, 50)

        spacecraft_orientation_history = {}
        for time, state in dynamics_simulator.state_history.items():
            spacecraft_orientation_history[time] = (
                quaternion_entries_to_rotation_matrix(state[6:10])
            )

        endurance_orientation_history = {}
        relative_distance = {}
        relative_speed = {}
        for (
            time,
            vars,
        ) in dynamics_simulator.propagation_results.dependent_variable_history.items():
            relative_distance[time] = vars[0]
            relative_speed[time] = vars[1]
            endurance_orientation_history[time] = vars[2:11].reshape(3, 3)

        cost = self.cost_function.cost(
            {
                "relative_distance": relative_distance,
                "relative_speed": relative_speed,
                "orientation_matrix_A": spacecraft_orientation_history,
                "orientation_matrix_B": endurance_orientation_history,
            }
        )

        del dynamics_simulator

        return [-1 * cost]

    def get_bounds(self):
        """Get bounds for neural network parameters (now with 3 outputs instead of 2)."""
        # Create a temporary controller to get the number of parameters
        simulator, _, controller, _ = self._create_simulator_components()
        num_params = len(controller.get_weights())

        # Set reasonable bounds for neural network weights
        lower_bounds = np.full(num_params, -10.0)  # Lower bound
        upper_bounds = np.full(num_params, 10.0)  # Upper bound

        return (lower_bounds, upper_bounds)


def main():
    """Test the stop neuron basic problem."""
    # Set loglevel to debug
    logging.basicConfig(
        level=logging.DEBUG, format="%(name)s - %(levelname)s - %(message)s"
    )
    # Suppress numba logging messages
    logging.getLogger("numba").setLevel(logging.WARNING)
    # Set seed for reproducible results
    np.random.seed(42)

    problem = StopNeuronBasicProblem(stop_threshold=0.7, max_simulation_time=30)

    # Get the number of parameters needed for the controller
    # Create temporary components to get controller info
    _, _, controller, _ = problem._create_simulator_components()
    num_params = len(controller.get_weights())
    print(f"Stop Neuron Controller requires {num_params} parameters")

    # Generate reproducible random weights
    import time

    t_start = time.time()
    # Evaluate fitness with the seeded random weights
    for i in range(3):
        random_weights = np.random.randn(num_params) * 0.1  # Small initial weights
        print(f"Iteration {i + 1}")
        fitness_value = problem.fitness(random_weights)
        print(f"Fitness with seeded random weights: {fitness_value}")

    print(f"Total time taken: {time.time() - t_start:.2f} seconds")
    return problem


if __name__ == "__main__":
    main()
