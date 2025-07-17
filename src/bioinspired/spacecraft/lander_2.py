"""Lander 2 spacecraft module.
This module defines the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
It is based on the paper
"""

import json
import numpy as np

from overrides import override
from numpy.linalg import norm

from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.numerical_simulation.propagation_setup import propagator
from tudatpy.numerical_simulation.propagation_setup import (
    acceleration,
    propagator,
)

from bioinspired.simulation.simulation_base import SimulatorBase
# from bioinspired.simulation import EmptyUniverseSimulator

from .spacecraft_base import SpacecraftBase


class Lander2(SpacecraftBase):
    """Lander 2 spacecraft design.

    This class implements the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
    It provides specific configurations and methods for the Lander 2 spacecraft.
    """

    def __init__(self, simulation: SimulatorBase, initial_state: np.ndarray):
        """Initialize the Lander 2 spacecraft with a name and initial state."""
        super().__init__("Lander 2", simulation, initial_state)
        # Load spacecraft configuration
        self._config = self._load_config()
        self._engines = self._config["engines"]
        self._physical_properties = self._config["physical_properties"]

        # Create convenient engine arrays
        self._engine_positions = np.array(
            [engine["location"] for engine in self._engines]
        )
        self._engine_directions = np.array(
            [engine["thrust_direction"] for engine in self._engines]
        )
        self._engine_max_thrusts = np.array(
            [engine["maximum_thrust"] for engine in self._engines]
        )
        self._engine_isps = np.array(
            [engine["specific_impulse"] for engine in self._engines]
        )

        # Create engine lookup dictionaries
        self._engines_by_id = {engine["id"]: engine for engine in self._engines}

        # self._simulation._get_body_model().get(self.name).constant_mass = self._physical_properties["dry_mass"] + self._physical_properties["fuel_mass"]
        self._simulation._get_body_model().get(self.name).mass = 10
        self.setup_engine_model()

    def _load_config(self) -> dict:
        """Load spacecraft configuration from JSON file."""
        with open("src/bioinspired/spacecraft/lander_2.json", "r") as file:
            return json.load(file)

    def get_engine_by_id(self, engine_id: str) -> dict:
        """Get engine configuration by ID."""
        return self._engines_by_id.get(engine_id)

    def get_engine_positions(self) -> np.ndarray:
        """Get all engine positions as a numpy array (N x 3)."""
        return self._engine_positions

    def get_engine_directions(self) -> np.ndarray:
        """Get all engine thrust directions as a numpy array (N x 3)."""
        return self._engine_directions

    def get_engine_max_thrusts(self) -> np.ndarray:
        """Get all engine maximum thrusts as a numpy array."""
        return self._engine_max_thrusts

    def get_inertia_tensor(self) -> np.ndarray:
        """Get the spacecraft inertia tensor as a numpy array."""
        return np.array(self._physical_properties["inertia_tensor"])

    def get_center_of_mass(self) -> np.ndarray:
        """Get the spacecraft center of mass as a numpy array."""
        return np.array(self._physical_properties["center_of_mass"])

    def get_dry_mass(self) -> float:
        """Get the spacecraft dry mass."""
        return self._physical_properties["dry_mass"]

    def get_fuel_mass(self) -> float:
        """Get the spacecraft fuel mass."""
        return self._physical_properties["fuel_mass"]

    def get_torque(self, control_vector: np.ndarray) -> np.ndarray:
        """Calculate the torque based on the control vector.
        Computed through sum of the moments due to RCS thrusters."""
        # Example implementation, replace with actual torque calculation logic.
        pass

    def get_thrust(self, current_time: float) -> np.ndarray:
        """Calculate the thrust based on the control vector.
        Computed through sum of the forces due to RCS thrusters."""
        # Example implementation, replace with actual thrust calculation logic.
        if not hasattr(self, "starttime"):
            self.starttime = current_time
        current_time = current_time - self.starttime if current_time is not None else 0
        # if current_time is not None:
            # print(f"Current time: {current_time}")
        thrust = np.zeros(3)
        thrust[0] = current_time/1000 if current_time is not None else 0
        thrust[1] = 10
        return thrust
        for engine, direction, max_thrust in zip(
            self._engines, self._engine_directions, self._engine_max_thrusts
        ):
            if engine["type"] == "rcs":
                thrust = control_vector * max_thrust
                return thrust * (direction / norm(direction))  # Normalize direction

    def get_thrust_direction(self, current_time: float) -> np.ndarray:
        """Calculate the thrust direction based on the control vector."""
        if current_time == current_time:
            thrust_vector = self.get_thrust(current_time)
            if norm(thrust_vector) == 0:
                return np.zeros(3)
            return thrust_vector / norm(thrust_vector)
        # If no computation is to be done, return zeros
        else:
            return np.zeros([3, 1])

    def get_thrust_magnitude(self, current_time: float) -> float:
        """Calculate the total thrust magnitude based on the control vector."""
        if current_time == current_time:
            thrust_vector = self.get_thrust(current_time)
            return norm(thrust_vector)
        # If no computation is to be done, return zeros
        else:
            return 0.0

    def setup_engine_model(self):
        """Setup the engine model for the spacecraft."""
        thrust_magnitude_settings = (
            propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(
                self.get_thrust_magnitude,
                self._engine_isps[0],
            )
        )
        environment_setup.add_engine_model(
            self.name,
            "ReactionControlSystem",
            thrust_magnitude_settings,
            self._simulation._get_body_model(),
        )

        # Create vehicle rotation model such that thrust points in required direction in inertial frame
        rotation_model_settings = (
            environment_setup.rotation_model.custom_inertial_direction_based(
                self.get_thrust_direction,
                self.name + "-fixed",
                self._simulation.global_frame_orientation,
            )
        )
        environment_setup.add_rotation_model(
            self._simulation._get_body_model(), self.name, rotation_model_settings
        )

    def _get_propagator(self) -> propagator.PropagatorSettings:
        """Return the propagator settings for the spacecraft."""
        # Create a propagator settings object for the spacecraft
        # Create propagation settings.
        return propagator.translational(
            self._simulation._get_central_body(),
            self._get_acceleration_model(),
            [self.name],
            self._initial_state,
            self._simulation._start_epoch,
            self._simulation._get_integrator(),
            self._get_termination(),
            # output_variables=dependent_variables_to_save
        )

    @override
    def _get_acceleration_settings(
        self,
    ) -> dict[str, dict[str, list[acceleration.AccelerationSettings]]]:
        """Compiles the acceleration model for point mass gravity from all bodies on the spacecraft, and adds RCS thrusters."""
        self._acceleration_settings = super()._get_acceleration_settings()
        self._acceleration_settings[self.name][self.name] = [
            propagation_setup.acceleration.thrust_from_engine("ReactionControlSystem")
        ]
        return self._acceleration_settings


# def main():
#     """Main function to demonstrate the usage of the Lander2 spacecraft."""
#     # Example usage of the Lander2 class.
#     simulation = (
#         EmptyUniverseSimulator()
#     )  # Assuming SimulatorBase is properly defined elsewhere.
#     initial_state = np.array([0, 0, 0, 0, 0, 0])  # Example initial state.

#     lander = Lander2("Lander2", simulation, initial_state)

#     # Demonstrate accessing the configuration data
#     print("=== Spacecraft Configuration ===")
#     print(f"Spacecraft name: {lander._config['spacecraft_name']}")
#     print(f"Dry mass: {lander.get_dry_mass()} kg")
#     print(f"Fuel mass: {lander.get_fuel_mass()} kg")
#     print(f"Center of mass: {lander.get_center_of_mass()}")

#     print("\n=== Inertia Tensor ===")
#     print(lander.get_inertia_tensor())

#     print("\n=== All Engines ===")
#     for i, engine in enumerate(lander._engines):
#         print(f"Engine {i}: {engine['id']} ({engine['type']})")
#         print(f"  Location: {engine['location']}")
#         print(f"  Thrust Direction: {engine['thrust_direction']}")
#         print(f"  Max Thrust: {engine['maximum_thrust']} N")


#     print("\n=== Engine Arrays ===")
#     print(f"Engine positions shape: {lander.get_engine_positions().shape}")
#     print(f"Engine directions shape: {lander.get_engine_directions().shape}")
#     print(f"Engine max thrusts: {lander.get_engine_max_thrusts()}")

#     print("\n=== Access by ID ===")
#     rcs_1 = lander.get_engine_by_id("rcs_1")
#     if rcs_1:
#         print(f"RCS_1 location: {rcs_1['location']}")
#         print(f"RCS_1 thrust: {rcs_1['maximum_thrust']} N")

#     print(f"\nInitialized spacecraft: {lander.name}")
#     print(f"Initial state: {lander._initial_state}")


# if __name__ == "__main__":
#     main()
