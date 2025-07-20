"""BioInspired - Lander 2 Playground Simulation"""

import numpy as np
from datetime import datetime, timezone

from tudatpy.astro.element_conversion import rotation_matrix_to_quaternion_entries

# Import the bioinspired modules
from bioinspired.simulation import EarthSimulator
from bioinspired.spacecraft import Lander2
from bioinspired.data import (
    init_db,
    save_simulation,
    save_spacecraft,
    save_trajectory,
    update_trajectory_status,
)

from bioinspired.controllers import MLPController


def main():
    print("=" * 60)
    print("BioInspired - Lander 2 Playground Simulation")
    print("=" * 60)

    # Initialize database (create tables if they don't exist)
    if not init_db():
        print("Failed to initialize database. Please check your PostgreSQL connection.")
        return

    # Create simulation
    print("\n2. Creating simulation...")
    simulator = EarthSimulator()

    # Save simulation to database
    print("\n3. Saving simulation to database...")
    sim_record = save_simulation(
        simulation=simulator,
        simulation_type="EarthSimulator",
    )
    print(f"[OK] Simulation saved with ID: {sim_record.id}")
    print(f"  Type: {sim_record.simulation_type}")

    # Create and save spacecraft
    print("\n4. Creating and saving spacecraft...")
    initial_state = np.array([6378e3, 0, 0, 0, 8e3, 0])  # Initial position (x, y, z)
    # Set initial rotation matrix (identity matrix)
    initial_rotation_matrix = np.eye(3)
    initial_rotational_velocity = np.array([0.01, 0, 0])  # Initial angular velocity (omega_x, omega_y, omega_z)
    # Set initial orientation by converting a rotation matrix to a Tudat-compatible quaternion
    initial_state_rotatation = rotation_matrix_to_quaternion_entries(initial_rotation_matrix)
    # Complete initial state by adding angular velocity vector (zero in this case)
    initial_state = np.concatenate((initial_state, initial_state_rotatation, initial_rotational_velocity))
    print(f"Initial state: {initial_state}")
    
    controller = MLPController(
        hidden_sizes=[64, 64],  # Two hidden layers with 64 neurons each
        output_size=3,  # Thrust vector in 3D space
        simulator=simulator,
        lander_name="Lander 2",
        target_name="Lander 2",
    )
    spacecraft = Lander2(initial_state=initial_state, simulation=simulator, controller=controller)

    craft_record = save_spacecraft(spacecraft=spacecraft, simulation_id=sim_record.id)
    print(f"[OK] Spacecraft saved with ID: {craft_record.id}")
    print(f"  Name: {craft_record.name}")
    print(f"  Type: {craft_record.spacecraft_type}")
    print(f"  Initial position: {craft_record.initial_state[:3]}")
    print(f"  Initial velocity: {craft_record.initial_state[3:6]}")

    # Create a trajectory for tracking this simulation execution with dynamics simulator
    print("\n5. Creating trajectory with dynamics simulator...")
    trajectory_record = save_trajectory(
        simulation_id=sim_record.id,
        spacecraft_id=craft_record.id,
        trajectory_metadata={
            "description": "Spacecraft trajectory in around Earth",
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

    # Run simulation (simplified example)
    print("\n7. Running simulation...")
    # try:
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=1000)
    print("[OK] Simulation completed successfully")

    # Update trajectory with completion info
    trajectory_record = update_trajectory_status(
        trajectory_id=trajectory_record.id,
        status="completed",
        completed_at=datetime.now(timezone.utc),
        dynamics_simulator=dynamics_simulator,
    )

    print("[OK] Trajectory updated with completion status")

    print("\n" + "=" * 60)
    print("Single Trajectory Simulation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
