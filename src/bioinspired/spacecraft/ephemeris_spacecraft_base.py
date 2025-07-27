"""Ephemeris Spacecraft Base Class
This module provides a base class for spacecraft that use ephemeris data for their trajectory.
After an initial setup phase, the spacecraft will follow a predefined trajectory based on ephemeris data.
"""

import numpy as np
from overrides import override
from abc import ABC, abstractmethod


from tudatpy.numerical_simulation.environment import SystemOfBodies
from tudatpy.numerical_simulation.propagation_setup import acceleration, propagator

from numpy.polynomial.chebyshev import chebpts2

from .spacecraft_base import SpacecraftBase
from .rotating_spacecraft_base import RotatingSpacecraftBase


class EphemerisSpacecraftBase(SpacecraftBase):
    """Base class for spacecraft that follow a trajectory defined by ephemeris data.

    A simulator and spacecraft are provided to the constructor.
    """

    def __init__(
        self,
        simulator,
        spacecraft: SpacecraftBase,
        initial_state,
        controller=None,
        
    ):
        """Initialize the ephemeris spacecraft with a simulator and spacecraft class
        Args:
            simulator (SimulationBase): The simulation environment.
            spacecraft (SpacecraftBase): The spacecraft to be used in the simulation.
        """
        self.simulator = simulator
        self.spacecraft = spacecraft
        # Check if spacecraft is or inherits from RotatingSpacecraftBase
        if isinstance(spacecraft, RotatingSpacecraftBase):
            if initial_state.shape != (13,):
                raise ValueError(
                    "Initial state for rotating spacecraft must be a 13-element vector representing position, velocity, quaternion orientation, and euler angular velocity."
                )
        else:
            if initial_state.shape != (6,):
                raise ValueError(
                    "Initial state must be a 6-element vector representing position and velocity."
                )
        super().__init__(
            name=spacecraft.name+"Ephemeris",
            simulation=simulator,
            initial_state=initial_state[:6]
        )
        self._initial_state = initial_state
        self.controller = controller

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
        chebyshev_points = chebpts2(n)
        # Scale points to the epoch range
        time_points = (
            0.5 * (end_epoch - start_epoch) * (chebyshev_points + 1) + start_epoch
        )

        simulator = type(self.simulator)()
        if self.controller is not None:
            spacecraft = type(self.spacecraft)(
                simulation=simulator, initial_state=self._initial_state
            )
        else:
            spacecraft = type(self.spacecraft)(
                simulation=simulator,
                initial_state=self._initial_state,
                controller=self.controller,
            )

        chebyshev_states = {}
        chebyshev_states[start_epoch] = self._initial_state
        for i in range(time_points.shape[0]-1):
            dynamics_simulator = simulator.run(time_points[i], time_points[i + 1])
            state_history = dynamics_simulator.state_history
            final_time = max(state_history.keys())
            final_state = state_history[final_time]
            chebyshev_states[time_points[i + 1]] = final_state

            del dynamics_simulator, spacecraft, simulator

            simulator = type(self.simulator)()
            if self.controller is not None:
                spacecraft = type(self.spacecraft)(
                    simulation=simulator, initial_state=final_state
                )
            else:
                spacecraft = type(self.spacecraft)(
                    simulation=simulator,
                    initial_state=final_state,
                    controller=self.controller,
                )
        return chebyshev_states

    def generate_ephemeris(self):
        """Generate the ephemeris data for the spacecraft.

        Args:
            ephemeris_data (np.ndarray): The ephemeris data to be used for the spacecraft trajectory.
        """
        # Check if the provided spacecraft is an instance of RotatingSpacecraftBase
        samples = self._create_chebychev_sampled_trajectory(
            start_epoch=self.simulator.start_epoch,
            end_epoch=self.simulator.end_epoch,
            n=self.simulator.n_chebyshev_points,
        )
        

    def _insert_into_body_model(self) -> SystemOfBodies:
        """Return the body model object."""
        if self._simulation.get_body_model().does_body_exist(self.name) is False:
            self._simulation.get_body_model().create_empty_body(self.name)
        else:
            raise ValueError(
                f"Body with name {self.name} already exists in the body model."
            )
        return self._simulation._body_model

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
