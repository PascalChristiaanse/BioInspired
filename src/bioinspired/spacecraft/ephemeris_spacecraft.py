"""Ephemeris Spacecraft Base Class
This module provides a base class for spacecraft that use ephemeris data for their trajectory.
After an initial setup phase, the spacecraft will follow a predefined trajectory based on ephemeris data.
"""

import numpy as np
from overrides import override
from numpy.polynomial.chebyshev import chebpts2

from tudatpy.numerical_simulation.environment_setup import (
    rotation_model,
    ephemeris,
    add_rotation_model,
    create_body_ephemeris,
)
from tudatpy.numerical_simulation.propagation_setup import acceleration, propagator
from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.astro.element_conversion import quaternion_entries_to_rotation_matrix
from tudatpy.math.interpolators import (
    lagrange_interpolation,
    create_one_dimensional_vector_interpolator,
    BoundaryInterpolationType,
)

from .spacecraft_base import SpacecraftBase
from .rotating_spacecraft_base import RotatingSpacecraftBase


class EphemerisSpacecraft(SpacecraftBase):
    """Base class for spacecraft that follow a trajectory defined by ephemeris data.

    A simulator and spacecraft are provided to the constructor.
    """

    def __init__(
        self,
        simulator,
        spacecraft_class: type[SpacecraftBase],
        initial_state,
        n_datapoints=2048,
        interpolator_order=10,
        controller=None,
    ):
        """Initialize the ephemeris spacecraft with a simulator and spacecraft class
        Args:
            simulator (SimulationBase): The simulation environment.
            spacecraft_class (type[SpacecraftBase]): The spacecraft class to be used in the simulation.
        """
        self.simulator = simulator
        self.spacecraft_class = spacecraft_class
        # Check if spacecraft class is or inherits from RotatingSpacecraftBase
        if issubclass(spacecraft_class, RotatingSpacecraftBase):
            if initial_state.shape != (13,):
                raise ValueError(
                    "Initial state for rotating spacecraft must be either a 13-element vector (position, velocity, quaternion orientation, euler angular velocity)."
                )
        else:
            if initial_state.shape != (6,):
                raise ValueError(
                    "Initial state must be a 6-element vector representing position and velocity."
                )
        super().__init__(
            name=spacecraft_class.__name__ + "-Ephemeris",
            simulation=simulator,
            initial_state=initial_state[:6],
        )
        self._initial_state = initial_state
        self.controller = controller

        # Determine if rotational state is available
        self.has_rotational_state = len(initial_state) > 6

        self.n_datapoints = n_datapoints
        self.interpolator_order = interpolator_order

        self.orientation_interpolator = None
        self.translational_interpolator = None

        self._current_orientation = quaternion_entries_to_rotation_matrix(
            self._initial_state[6:10]
        )

    def _create_chebychev_sampled_trajectory(self, start_epoch, end_epoch, n):
        """Create a Chebyshev sampled trajectory to be used in an interpolator.
        Args:
            start_epoch (float): Start epoch of the trajectory.
            end_epoch (float): End epoch of the trajectory.
            n (int): Number of Chebyshev points to sample.
        Returns:
            dict[np.ndarray]: Chebyshev sampled trajectory.
        """

        # Create Chebyshev points
        chebyshev_points = chebpts2(n)  # Scale points to the epoch range
        time_points = (
            0.5 * (end_epoch - start_epoch) * (chebyshev_points + 1) + start_epoch
        )

        simulator = type(self.simulator)()
        if self.controller is None:
            spacecraft = self.spacecraft_class(
                simulation=simulator, initial_state=self._initial_state
            )
        else:
            spacecraft = self.spacecraft_class(
                simulation=simulator,
                initial_state=self._initial_state,
                controller=self.controller,
            )

        chebyshev_states = {}
        chebyshev_states[start_epoch] = self._initial_state
        for i in range(time_points.shape[0] - 1):
            dynamics_simulator = simulator.run(time_points[i], time_points[i + 1])
            state_history = dynamics_simulator.state_history
            final_time = max(state_history.keys())
            final_state = state_history[final_time]
            chebyshev_states[time_points[i + 1]] = final_state

            del dynamics_simulator, spacecraft, simulator

            simulator = type(self.simulator)()
            if self.controller is None:
                spacecraft = self.spacecraft_class(
                    simulation=simulator, initial_state=final_state
                )
            else:
                spacecraft = self.spacecraft_class(
                    simulation=simulator,
                    initial_state=final_state,
                    controller=self.controller,
                )
        return chebyshev_states

    def load_ephemeris(self, file_path: str = None):
        """Load ephemeris data, either from file or by generating one.

        Args:
            file_path (str): Path to the ephemeris data file. If None, generates a new ephemeris.
        """
        if file_path is not None:
            # Load ephemeris data from file
            import pickle

            with open(file_path, "rb") as f:
                state_dict = pickle.load(f)
        else:
            # Generate ephemeris data
            state_dict = self._generate_ephemeris_data()
        # Create the ephemeris from the state dictionary
        self._create_ephemeris(state_dict)

    def _format_file_name(self):
        """Creates a file name for the ephemeris data based on the spacecraft name and current date."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.spacecraft_class.__name__}_ephemeris_{timestamp}.pkl"

    def save_ephemeris(self, file_path, file_name=None):
        """Save the current ephemeris data to a file.

        Args:
            file_path (str): Path to the file where the ephemeris data will be saved.
            file_name (str, optional): Name of the file to save the ephemeris data. If None, uses a default format.
        """
        # Implement saving logic here
        import pickle

        if file_name is None:
            file_name = self._format_file_name()
        full_path = f"{file_path}/{file_name}"

        state_dict = self._generate_ephemeris_data()
        with open(full_path, "wb") as f:
            pickle.dump(state_dict, f)

    def _create_ephemeris(self, state_dict=None):
        """Create the ephemeris for the spacecraft."""
        # Generate the ephemeris data
        state_dict = self._generate_ephemeris_data()

        # Translational interpolator
        translational_settings = lagrange_interpolation(
            self.interpolator_order,
            boundary_interpolation=BoundaryInterpolationType.use_boundary_value,
        )
        self.translational_interpolator = create_one_dimensional_vector_interpolator(
            state_dict["translational_state"],
            translational_settings,
        )

        # Orientation interpolator (only if rotational state is available)
        if self.has_rotational_state and state_dict["orientation"] is not None:
            try:
                orientation_settings = lagrange_interpolation(
                    self.interpolator_order,
                    boundary_interpolation=BoundaryInterpolationType.use_boundary_value,
                )
                self.orientation_interpolator = (
                    create_one_dimensional_vector_interpolator(
                        state_dict["orientation"],
                        orientation_settings,
                    )
                )
            except Exception as e:
                print(f"Warning: Could not create orientation interpolator: {e}")
                self.orientation_interpolator = None

    def _generate_ephemeris_data(self):
        """Generate the ephemeris data for the spacecraft.

        Args:
            ephemeris_data (np.ndarray): The ephemeris data to be used for the spacecraft trajectory.
        """
        # Check if the provided spacecraft is an instance of RotatingSpacecraftBase
        samples = self._create_chebychev_sampled_trajectory(
            start_epoch=self.simulator._start_epoch,
            end_epoch=self.simulator._end_epoch + 50,
            n=self.n_datapoints * 1.5,
        )
        state_dict = self._split_state(samples)
        return state_dict

    def _split_state(self, state):
        """Split the state vector into position, velocity, and orientation components."""
        position = {}
        velocity = {}
        translational_state = {}
        orientation = {}
        angular_velocity = {}

        has_orientation_data = False

        for time, values in state.items():
            position[time] = values[:3]
            velocity[time] = values[3:6]
            translational_state[time] = np.concatenate([values[:3], values[3:6]])
            if len(values) > 6:
                orientation[time] = values[6:10]
                angular_velocity[time] = values[10:13]
                has_orientation_data = True

        # Set to None if no orientation data was found
        if not has_orientation_data:
            orientation = None
            angular_velocity = None

        return {
            "position": position,
            "velocity": velocity,
            "translational_state": translational_state,
            "orientation": orientation,
            "angular_velocity": angular_velocity,
        }

    def _insert_into_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        body_model = self._simulation.get_body_model()
        if body_model.does_body_exist(self.name) is False:
            body_model.create_empty_body(self.name)
        else:
            raise ValueError(
                f"Body with name {self.name} already exists in the body model."
            )

        # Add custom ephemeris data to the body model

        # Translation model settings for the spacecraft
        translational_model_settings = ephemeris.custom_ephemeris(
            self._state_function,
            self.simulator.global_frame_origin,
            self.simulator.global_frame_orientation,
        )
        translational_ephemeris = create_body_ephemeris(
            translational_model_settings, self.name
        )
        body_model.get(self.name).ephemeris = translational_ephemeris

        # Rotation model settings for the spacecraft (only if rotational state is available)
        try:
            rotation_model_settings = rotation_model.custom_rotation_model(
                self.simulator.global_frame_orientation,
                self.name + "-Fixed",
                self._rotation_matrix_function,
                1e-2,
            )
            add_rotation_model(
                body_model,
                self.name,
                rotation_model_settings,
            )
        except Exception as e:
            print(f"Warning: Could not add rotation model for {self.name}: {e}")
            # Continue without rotation model - spacecraft will use default orientation

        return self._simulation._body_model

    def get_orientation(self) -> np.ndarray:
        """Get the orientation of the spacecraft at a given time.

        Returns:
            np.ndarray: Orientation matrix of size 9 body-fixed to inertial frame.
        """
        return self._current_orientation.flatten()

    @override
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft."""
        # Create global accelerations dictionary.
        return {}

    def _get_propagator(self) -> list[propagator.PropagatorSettings]:
        """Return the propagator settings for the spacecraft."""
        # Create a propagator settings object for the spacecraft
        return []

    def _rotation_matrix_function(self, time: float) -> np.ndarray:
        """Computes the rotation matrix for the spacecraft at a given time based on the orientation interpolator.

        Returns an identity matrix if no orientation data is available.
        """
        if self.orientation_interpolator is None:
            # Return 3x3 identity matrix when no rotational state is available
            return np.eye(3)

        try:
            quaternion = self.orientation_interpolator.interpolate(time)
            self._current_orientation = quaternion_entries_to_rotation_matrix(
                quaternion
            )
            return self._current_orientation
        except Exception as e:
            print(f"Warning: Could not interpolate orientation at time {time}: {e}")
            return np.eye(3)

    def _state_function(self, time: float) -> np.ndarray:
        """Computes the state vector for the spacecraft at a given time based on the translational interpolator."""
        if self.translational_interpolator is None:
            raise ValueError("Translational interpolator is not set.")

        try:
            return self.translational_interpolator.interpolate(time)
        except Exception as e:
            raise ValueError(
                f"Could not interpolate translational state at time {time}: {e}"
            )
