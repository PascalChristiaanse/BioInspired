"""Basic problem module.
The basic problem is defined as:
 - In plane (XY+R_Z) problem
 - Two thrusters
 - Empty universe
 - No rotation on target
 - 6 input neurons based on J. Leitner (2010)
 - 10 hidden neurons
 - 2 + 0 output neurons (no stop neuron)
"""

import numpy as np
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
from bioinspired.problem import JLeitner2010NoStopNeuron

from bioinspired.spacecraft import Lander2, Endurance, EphemerisSpacecraft
from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.controller import MLPController


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


class BasicController(MLPController):
    """Basic controller for the automatic rendezvous and docking (AR&D) problem."""

    def __init__(
        self,
        simulator,
        lander_name,
        target_name,
        **kwargs,
    ):
        """Initialize the basic controller."""
        super().__init__(
            6,
            [10],
            2,
            simulator,
            lander_name,
            target_name,
            **kwargs,
        )

    @override
    def extract_state_vector(self, current_time) -> np.ndarray:
        lander_body = self.simulator.get_body_model().get(self.lander_name)
        lander_state = (
            lander_body.state if hasattr(lander_body, "state") else np.zeros(6)
        )

        target_body = self.simulator.get_body_model().get(self.target_name)
        target_state = target_body.ephemeris.cartesian_state(current_time)

        x_diff = target_state[0] - lander_state[0]
        y_diff = target_state[1] - lander_state[1]
        x_dot_diff = target_state[3] - lander_state[3]
        y_dot_diff = target_state[4] - lander_state[4]

        lander_orientation = getattr(
            lander_body, "body_fixed_to_inertial_frame", np.zeros(9)
        )
        lander_z_angle = rotation_matrix_to_euler_angles(lander_orientation)[2]

        target_orientation = target_body.rotation_model.body_fixed_to_inertial_rotation(
            current_time
        )
        target_z_angle = rotation_matrix_to_euler_angles(target_orientation)[2]
        z_angle_diff = target_z_angle - lander_z_angle

        lander_angular_rate = rotation_matrix_to_euler_angles(
            getattr(lander_body, "inertial_to_body_fixed_frame", np.zeros(3))
        )

        target_angular_rate = (
            target_body.rotation_model.time_derivative_body_fixed_to_inertial_rotation(
                current_time
            )
        )
        z_rate_diff = (lander_angular_rate[2] - target_angular_rate[2])[2]

        return np.array(
            [x_diff, y_diff, x_dot_diff, y_dot_diff, z_angle_diff, z_rate_diff]
        )

    def get_control_action(self, current_time):
        """Return the modified control action for two thrusters."""
        state_vector = self.extract_state_vector(current_time)
        control_vector = self.forward(state_vector)
        control_vector = control_vector.detach().numpy()
        # Modify the thrust vector for two thrusters
        modified_thrust_vector = np.zeros(24)
        modified_thrust_vector[8] = control_vector[0]
        modified_thrust_vector[13] = control_vector[1]
        return modified_thrust_vector


def test() -> np.ndarray:
    return np.ones(3)


class BasicProblem(ProblemBase):
    """Basic problem class for the automatic rendezvous and docking (AR&D) problem."""

    def __init__(self):
        super().__init__(JLeitner2010NoStopNeuron())
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

        controller = BasicController(
            simulator=simulator,
            lander_name="Lander 2",
            target_name="Endurance-Ephemeris",
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

        dynamics_simulator = simulator.run(0, 10)

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
        return [-1 * cost]

    def get_bounds(self):
        """Get bounds for 92 neural network parameters."""
        # Create a temporary controller to get the number of parameters
        simulator, _, controller, _ = self._create_simulator_components()
        num_params = len(controller.get_weights())

        # Set reasonable bounds for neural network weights
        lower_bounds = np.full(num_params, -5.0)  # Lower bound
        upper_bounds = np.full(num_params, 5.0)  # Upper bound

        return (lower_bounds, upper_bounds)


def main():
    # Set seed for reproducible results
    np.random.seed(42)

    problem = BasicProblem()

    # Get the number of parameters needed for the controller
    # Create temporary components to get controller info
    _, _, controller, _ = problem._create_simulator_components()
    num_params = len(controller.get_weights())
    print(f"Controller requires {num_params} parameters")

    # Generate reproducible random weights
    random_weights = np.random.randn(num_params) * 0.1  # Small initial weights

    # Evaluate fitness with the seeded random weights
    fitness_value = problem.fitness(random_weights)
    print(f"Fitness with seeded random weights: {fitness_value}")

    return problem, fitness_value


if __name__ == "__main__":
    main()
