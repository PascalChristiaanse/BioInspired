"""Lander 2 spacecraft module.
This module defines the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
It is based on the paper
"""

import json
import numpy as np

from tudatpy.numerical_simulation.propagation_setup import propagator
from bioinspired.simulation.simulation_base import SimulatorBase
from bioinspired.simulation.empty_universe import EmptyUniverseSimulator

from .spacecraft_base import SpacecraftBase


class Lander2(SpacecraftBase):
    """Lander 2 spacecraft design.

    This class implements the Lander 2 spacecraft design, inheriting from the SpacecraftBase class.
    It provides specific configurations and methods for the Lander 2 spacecraft.
    """

    def __init__(self, name: str, simulation: SimulatorBase, initial_state: np.ndarray):
        """Initialize the Lander 2 spacecraft with a name and initial state."""
        super().__init__(name, simulation, initial_state)
        # Load spacecraft configuration
        self._config = self._load_config()
        self._engines = self._config["engines"]
        self._physical_properties = self._config["physical_properties"]
        
        # Create convenient engine arrays
        self._engine_positions = np.array([engine["location"] for engine in self._engines])
        self._engine_directions = np.array([engine["thrust_direction"] for engine in self._engines])
        self._engine_max_thrusts = np.array([engine["maximum_thrust"] for engine in self._engines])
        self._engine_isps = np.array([engine["specific_impulse"] for engine in self._engines])
        
        # Create engine lookup dictionaries
        self._engines_by_id = {engine["id"]: engine for engine in self._engines}
        self._engines_by_type = {}
        for engine in self._engines:
            engine_type = engine["type"]
            if engine_type not in self._engines_by_type:
                self._engines_by_type[engine_type] = []
            self._engines_by_type[engine_type].append(engine)

    def _load_config(self) -> dict:
        """Load spacecraft configuration from JSON file."""
        with open("src/bioinspired/spacecraft/lander_2.json", "r") as file:
            return json.load(file)
    
    def _load_engine_settings(self) -> dict:
        """Load engine settings from a JSON file. (Deprecated - use _load_config instead)"""
        return self._load_config()
    
    def get_engine_by_id(self, engine_id: str) -> dict:
        """Get engine configuration by ID."""
        return self._engines_by_id.get(engine_id)
    
    def get_engines_by_type(self, engine_type: str) -> list:
        """Get all engines of a specific type (e.g., 'rcs', 'main_propulsion')."""
        return self._engines_by_type.get(engine_type, [])
    
    def get_rcs_engines(self) -> list:
        """Get all RCS engines."""
        return self.get_engines_by_type("rcs")
    
    def get_main_engines(self) -> list:
        """Get all main propulsion engines."""
        return self.get_engines_by_type("main_propulsion")
    
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

    def get_thrust(self, control_vector: np.ndarray) -> np.ndarray:
        """Calculate the thrust based on the control vector.
        Computed through sum of the forces due to RCS thrusters."""
        # Example implementation, replace with actual thrust calculation logic.
        pass

    def _get_acceleration_settings(self):
        return super()._get_acceleration_settings()

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


def main():
    """Main function to demonstrate the usage of the Lander2 spacecraft."""
    # Example usage of the Lander2 class.
    simulation = (
        EmptyUniverseSimulator()
    )  # Assuming SimulatorBase is properly defined elsewhere.
    initial_state = np.array([7000, 0, 0, 0, 7.12, 0])  # Example initial state.

    lander = Lander2("Lander2", simulation, initial_state)
    
    # Demonstrate accessing the configuration data
    print("=== Spacecraft Configuration ===")
    print(f"Spacecraft name: {lander._config['spacecraft_name']}")
    print(f"Dry mass: {lander.get_dry_mass()} kg")
    print(f"Fuel mass: {lander.get_fuel_mass()} kg")
    print(f"Center of mass: {lander.get_center_of_mass()}")
    
    print("\n=== Inertia Tensor ===")
    print(lander.get_inertia_tensor())
    
    print("\n=== All Engines ===")
    for i, engine in enumerate(lander._engines):
        print(f"Engine {i}: {engine['id']} ({engine['type']})")
        print(f"  Location: {engine['location']}")
        print(f"  Thrust Direction: {engine['thrust_direction']}")
        print(f"  Max Thrust: {engine['maximum_thrust']} N")
    
    print("\n=== RCS Engines Only ===")
    rcs_engines = lander.get_rcs_engines()
    print(f"Number of RCS engines: {len(rcs_engines)}")
    for engine in rcs_engines:
        print(f"  {engine['id']}: {engine['maximum_thrust']} N at {engine['location']}")
    
    print("\n=== Main Engines Only ===")
    main_engines = lander.get_main_engines()
    for engine in main_engines:
        print(f"  {engine['id']}: {engine['maximum_thrust']} N at {engine['location']}")
    
    print("\n=== Engine Arrays ===")
    print(f"Engine positions shape: {lander.get_engine_positions().shape}")
    print(f"Engine directions shape: {lander.get_engine_directions().shape}")
    print(f"Engine max thrusts: {lander.get_engine_max_thrusts()}")
    
    print("\n=== Access by ID ===")
    rcs_1 = lander.get_engine_by_id("rcs_1")
    if rcs_1:
        print(f"RCS_1 location: {rcs_1['location']}")
        print(f"RCS_1 thrust: {rcs_1['maximum_thrust']} N")
    
    print(f"\nInitialized spacecraft: {lander.name}")
    print(f"Initial state: {lander._initial_state}")


if __name__ == "__main__":
    main()
