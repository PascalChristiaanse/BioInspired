"""Run a single trajectory simulation and integrate with the BioInspired database.
"""

import numpy as np
from datetime import datetime, timezone

# Import the bioinspired modules
from bioinspired.simulation import EmptyUniverseSimulator
from bioinspired.simulation import EarthSimulator
from bioinspired.spacecraft.simple_craft import SimpleCraft
from bioinspired.data import (
    init_db,
    save_simulation,
    save_spacecraft,
    save_trajectory,
    update_trajectory_status,
    get_simulation,
    get_spacecraft_by_simulation,
    get_trajectories_by_simulation,
    get_simulation_status,
    get_trajectories_by_spacecraft,
)


def main():

    print("=" * 60)
    print("BioInspired - Single Trajectory Simulation")
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
        simulation_type="EarthUniverse",
    )
    print(f"[OK] Simulation saved with ID: {sim_record.id}")
    print(f"  Type: {sim_record.simulation_type}")

    # Create and save spacecraft
    print("\n4. Creating and saving spacecraft...")
    initial_state = np.array([6378e3, 0, 0, 0, 10e3, 0])  # [x, y, z, vx, vy, vz]
    spacecraft = SimpleCraft(initial_state=initial_state, simulation=simulator)

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
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=3600*6)
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
