"""
Example script demonstrating how to save simulation and spacecraft data
to the PostgreSQL database using the updated data module.
"""

import numpy as np
from datetime import datetime, timezone

# Import the bioinspired modules
from bioinspired.simulation.empty_universe import EmptyUniverseSimulator
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
)


def main():
    """Main function to demonstrate database integration."""
    print("=" * 60)
    print("BioInspired Database Integration Example")
    print("=" * 60)

    # Initialize database (create tables if they don't exist)
    print("\n1. Initializing database...")
    if not init_db():
        print("Failed to initialize database. Please check your PostgreSQL connection.")
        return

    # Create simulation
    print("\n2. Creating simulation...")
    simulator = EmptyUniverseSimulator()

    # Save simulation to database
    print("\n3. Saving simulation to database...")
    sim_record = save_simulation(
        simulation=simulator,
        simulation_type="EmptyUniverseSimulator",
    )
    print(f"[OK] Simulation saved with ID: {sim_record.id}")
    print(f"  Type: {sim_record.simulation_type}")

    # Create and save spacecraft
    print("\n4. Creating and saving spacecraft...")
    initial_state = np.array([6378e3, 0, 0, 0, 8e3, 0])  # [x, y, z, vx, vy, vz]
    spacecraft = SimpleCraft(initial_state=initial_state, simulation=simulator)

    craft_record = save_spacecraft(spacecraft=spacecraft, simulation_id=sim_record.id)
    print(f"[OK] Spacecraft saved with ID: {craft_record.id}")
    print(f"  Name: {craft_record.name}")
    print(f"  Type: {craft_record.spacecraft_type}")
    print(f"  Initial position: {craft_record.initial_state[:3]}")
    print(f"  Initial velocity: {craft_record.initial_state[3:6]}")

    # Display termination settings if available (commented out until schema is updated)
    # if craft_record.termination_settings is not None:
    #     print("  Spacecraft termination settings available")
    # if sim_record.termination_settings is not None:
    #     print("  Simulation termination settings available")

    # Create a trajectory for tracking this simulation execution with dynamics simulator
    print("\n5. Creating trajectory with dynamics simulator...")
    trajectory_record = save_trajectory(
        simulation_id=sim_record.id,
        trajectory_metadata={
            "description": "Spacecraft trajectory in empty universe",
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
    dynamics_simulator = simulator.run(start_epoch=0.0, simulation_time=100.0)
    print("[OK] Simulation completed successfully")

    # Update trajectory with completion info
    trajectory_record = update_trajectory_status(
        trajectory_id=trajectory_record.id,
        status="completed",
        completed_at=datetime.now(timezone.utc),
        dynamics_simulator=dynamics_simulator,
    )

    # Note: save_simulation_result is deprecated, but still works for backward compatibility
    print("[OK] Trajectory updated with completion status")

    # except Exception as e:
    #     print(f"[ERROR] Simulation failed: {e}")
    #     update_trajectory_status(
    #         trajectory_id=trajectory_record.id, status="failed", error_message=str(e)
    #     )

    # Demonstrate the new dump methods for configuration data
    print("\n4.5. Demonstrating dump methods...")
    try:
        # Show simulation dump data
        body_model_data = simulator._dump_body_model()
        print(f"[OK] Body model dump: {len(body_model_data)} characters")

        integrator_data = simulator._dump_integrator_settings()
        print(f"[OK] Integrator settings dump: {len(integrator_data)} characters")

        sim_termination_data = simulator.dump_termination_conditions()
        print(f"[OK] Simulation termination dump: {len(sim_termination_data)} characters")

        # Show spacecraft dump data
        accel_data = spacecraft._dump_acceleration_settings()
        print(f"[OK] Acceleration settings dump: {len(accel_data)} characters")

        craft_termination_data = spacecraft.dump_termination_settings()
        print(
            f"[OK] Spacecraft termination dump: {len(craft_termination_data)} characters"
        )

    except Exception as e:
        print(f"[ERROR] Error demonstrating dump methods: {e}")

    # Demonstrate termination condition management
    print("\n4.6. Setting up termination conditions...")
    try:
        # Add simulation-level termination condition (time limit)
        simulator.add_termination_condition(
            {
                "type": "propagator.PropagationTimeTerminationSettings",
                "condition": None,
                "value": 100.0,  # 100 seconds
            }
        )
        print("[OK] Added simulation time termination condition (100s)")

        # Add spacecraft-level termination condition (example)
        spacecraft.add_termination_condition(
            {
                "type": "propagator.PropagationTimeTerminationSettings",
                "condition": None,
                "value": 50.0,  # 50 seconds (will trigger first)
            }
        )
        print("[OK] Added spacecraft time termination condition (50s)")
        print("  Note: During simulation, conditions from both levels are merged")

    except Exception as e:
        print(f"[ERROR] Error setting termination conditions: {e}")

    # Retrieve and display saved data
    print("\n8. Retrieving saved data...")
    retrieved_sim = get_simulation(sim_record.id)
    if retrieved_sim:
        print(f"[OK] Retrieved simulation ID: {retrieved_sim.id}")
        print(f"  Type: {retrieved_sim.simulation_type}")

        # Get simulation status from trajectory
        sim_status = get_simulation_status(retrieved_sim.id)
        print(f"  Status: {sim_status}")

    spacecraft_list = get_spacecraft_by_simulation(sim_record.id)
    print(f"[OK] Found {len(spacecraft_list)} spacecraft in simulation")
    for sc in spacecraft_list:
        print(f"  - {sc.name} ({sc.spacecraft_type})")

    # Show trajectory information
    trajectories = get_trajectories_by_simulation(sim_record.id)
    print(f"[OK] Found {len(trajectories)} trajectories for simulation")
    for traj in trajectories:
        print(f"  - Trajectory {traj.id}: {traj.status})")
        print(f"    Data size: {traj.data_size} points")
        print(f"    Created: {traj.created_at}")
        if traj.started_at is not None:
            print(f"    Started: {traj.started_at}")
        if traj.completed_at is not None:
            print(f"    Completed: {traj.completed_at}")

    print("\n" + "=" * 60)
    print("Database integration example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
