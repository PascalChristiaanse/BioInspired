"""Full simulator playground.
This app implements:
 - Lander 2 spacecraft simulation
    - Thrust control with ConstantController
 - Endurance spacecraft simulation
    - Full simulation
    - Ephemeris based simulation
 - Trajectory management with PostgreSQL
"""

import numpy as np
from datetime import datetime, timezone

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries
from tudatpy.numerical_simulation.propagation_setup import dependent_variable
from tudatpy.numerical_simulation.propagation import (
    create_dependent_variable_dictionary,
)

# Import the bioinspired modules
from bioinspired.simulation import EarthSimulator
from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.spacecraft import Lander2, Endurance, EphemerisSpacecraft
from bioinspired.data import (
    init_db,
    save_simulation,
    save_spacecraft,
    save_trajectory,
    update_trajectory_status,
)

from bioinspired.controller import MLPController
from bioinspired.controller import ConstantController


def main():
    # Initialize database (create tables if they don't exist)
    if not init_db():
        print("Failed to initialize database. Please check your PostgreSQL connection.")
        return

    print("\n1. Creating simulation with ephemeris spacecraft...")
    simulator = EmptyUniverseSimulator(
        dependent_variables_list=[
            dependent_variable.relative_position("Lander 2", "Endurance-Ephemeris")
        ]
    )

    # Save simulation to database
    print("\n2. Saving simulation to database...")
    sim_record = save_simulation(
        simulation=simulator,
        simulation_type="EmptyUniverseSimulator",
    )
    print(f"[OK] Simulation saved with ID: {sim_record.id}")
    print(f"  Type: {sim_record.simulation_type}")

    # Create ephemeris spacecraft (Endurance)
    print("\n3. Creating Endurance ephemeris spacecraft...")
    endurance = EphemerisSpacecraft(
        simulator, Endurance, np.array([200, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -7])
    )
    endurance.load_ephemeris()
    print("[OK] Endurance ephemeris loaded")

    # Create and save Lander 2 spacecraft
    print("\n4. Creating and saving Lander 2 spacecraft...")
    translational_state = np.array([0, 0, 0, 0, 0, 0])  # Initial position (x, y, z)
    orientation = np.eye(3)
    angular_velocity = np.array([0.0, 0, 0])
    orientation_matrix = rotation_matrix_to_quaternion_entries(orientation)

    initial_state = np.concatenate(
        (translational_state, orientation_matrix, angular_velocity)
    )
    print(f"  Initial state: {initial_state}")

    controller = MLPController(
        hidden_sizes=[10, 10],
        output_size=24,
        simulator=simulator,
        lander_name="Lander 2",
        target_name="Endurance-Ephemeris",
    )
    spacecraft = Lander2(
        initial_state=initial_state, simulation=simulator, controller=controller
    )

    craft_record = save_spacecraft(spacecraft=spacecraft, simulation_id=sim_record.id)
    print(f"[OK] Spacecraft saved with ID: {craft_record.id}")
    print(f"  Name: {craft_record.name}")
    print(f"  Type: {craft_record.spacecraft_type}")
    print(f"  Initial position: {craft_record.initial_state[:3]}")
    print(f"  Initial velocity: {craft_record.initial_state[3:6]}")

    # Create a trajectory for tracking this simulation execution
    print("\n5. Creating trajectory...")
    trajectory_record = save_trajectory(
        simulation_id=sim_record.id,
        spacecraft_id=craft_record.id,
        trajectory_metadata={
            "description": "Lander 2 spacecraft trajectory relative to Endurance ephemeris",
            "controller_type": "MLPController",
            "target_spacecraft": "Endurance-Ephemeris",
        },
    )
    print(f"[OK] Trajectory created with ID: {trajectory_record.id}")

    # Update trajectory status to running
    print("\n6. Updating trajectory status to running...")
    trajectory_record = update_trajectory_status(
        trajectory_id=trajectory_record.id,
        status="running",
        started_at=datetime.now(timezone.utc),
    )
    print(f"[OK] Trajectory status: {trajectory_record.status}")

    # Run simulation
    print("\n7. Running simulation...")
    try:
        dynamics_simulator = simulator.run(0, 100)
        print("[OK] Simulation completed successfully")
        
        # Extract and display results
        final_time = max(dynamics_simulator.state_history.keys())
        history_dict = create_dependent_variable_dictionary(dynamics_simulator)
        position_history = history_dict[
            dependent_variable.relative_position("Lander 2", "Endurance-Ephemeris")
        ]
        print(f"  Final time: {final_time}")
        print(f"  Position history entries: {len(position_history)}")
        
        # Update trajectory with completion info
        trajectory_record = update_trajectory_status(
            trajectory_id=trajectory_record.id,
            status="completed",
            completed_at=datetime.now(timezone.utc),
            dynamics_simulator=dynamics_simulator,
        )
        print("[OK] Trajectory updated with completion status")
        
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        # Update trajectory with error status
        trajectory_record = update_trajectory_status(
            trajectory_id=trajectory_record.id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
        )
        print("[ERROR] Trajectory updated with failure status")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("BioInspired - Simulator Playground")
    print("=" * 60)
    main()
    print("\n" + "=" * 60)
    print("Full Simulator Playground completed successfully!")
    print("=" * 60)
