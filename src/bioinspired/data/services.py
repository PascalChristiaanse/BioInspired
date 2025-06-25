"""
Database service functions for simulations and spacecraft.
This module provides high-level functions to save and retrieve
simulation and spacecraft data from the PostgreSQL database.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from sqlalchemy import cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session

from .database import get_session_context
from .models import Simulation, Spacecraft, Trajectory


# try:
#     from bioinspired.simulation.simulation_base import SimulatorBase
#     from bioinspired.spacecraft.spacecraft_base import SpacecraftBase
# except ImportError as e:
#     from src.bioinspired.simulation.simulation_base import SimulatorBase
#     from src.bioinspired.spacecraft.spacecraft_base import SpacecraftBase
# except Exception as e:
#     raise ImportError(
#         "Ensure the 'bioinspired' package is installed and accessible. "
#         f"Error: {e}"
#     )

def serialize_numpy_array(arr: np.ndarray) -> List[float]:
    """Convert numpy array to JSON-serializable list."""
    return arr.flatten().tolist()


def serialize_dynamics_simulator(dynamics_simulator):
    """
    Serialize a dynamics simulator object to a JSON-serializable dict.
    Handles numpy arrays, tuple keys, and enums.
    """
    import numpy as np

    def convert_keys_to_str(d):
        # Recursively convert all dict keys to strings
        if isinstance(d, dict):
            return {str(k): convert_keys_to_str(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [convert_keys_to_str(x) for x in d]
        elif isinstance(d, np.ndarray):
            return d.tolist()
        elif hasattr(d, "name") and hasattr(d, "value"):
            # Enum
            return str(d)
        else:
            return d

    if dynamics_simulator is None:
        return None

    pr = dynamics_simulator.propagation_results
    td = pr.termination_details

    dynamics_simulator_dict = {
        "integration_completed_successfully": dynamics_simulator.integration_completed_successfully,
        "cumulative_computation_time_history": convert_keys_to_str(
            pr.cumulative_computation_time_history
        ),
        "cumulative_number_of_function_evaluations": convert_keys_to_str(
            pr.cumulative_number_of_function_evaluations_history
        ),
        "dependent_variable_history": convert_keys_to_str(
            pr.dependent_variable_history
        ),
        "dependent_variable_ids": convert_keys_to_str(pr.dependent_variable_ids),
        "initial_and_final_times": list(pr.initial_and_final_times)
        if hasattr(pr, "initial_and_final_times")
        else None,
        "processed_state_ids": convert_keys_to_str(pr.processed_state_ids),
        "propagated_state_ids": convert_keys_to_str(pr.propagated_state_ids),
        "propagated_state_vector_length": pr.propagated_state_vector_length,
        "state_history": convert_keys_to_str(pr.state_history),
        "termination_details": {
            "terminated_on_exact_condition": td.terminated_on_exact_condition,
            "termination_reason": str(td.termination_reason)
            if hasattr(td, "termination_reason")
            else None,
            "was_condition_met_when_stopping": td.was_condition_met_when_stopping,
        },
        "total_computation_time": pr.total_computation_time,
        "total_number_of_function_evaluations": pr.total_number_of_function_evaluations,
    }

    return dynamics_simulator_dict


def deserialize_numpy_array(
    data: List[float], shape: Optional[tuple] = None
) -> np.ndarray:
    """Convert list back to numpy array with optional reshaping."""
    arr = np.array(data)
    if shape:
        arr = arr.reshape(shape)
    return arr


def save_simulation(
    simulation,
    simulation_type: str,
    session: Optional[Session] = None,
) -> Simulation:
    """
    Save a simulation configuration to the database.
    If an identical simulation already exists, return that one.

    Args:
        simulation: The simulator instance
        simulation_type: Type of simulation (e.g., 'EmptyUniverseSimulator')
        session: Optional database session (will create one if not provided)

    Returns:
        The created or existing Simulation database record
    """

    def _save_sim(session: Session) -> Simulation:
        # Use new dump methods for better serialization
        try:
            integrator_settings = json.loads(simulation._dump_integrator_settings())
        except (AttributeError, json.JSONDecodeError):
            # Fallback to old method if dump method fails
            integrator_settings = {}

        try:
            body_model_settings = json.loads(simulation._dump_body_model())
        except (AttributeError, json.JSONDecodeError):
            # Fallback to old method if dump method fails
            body_model_settings = {}
            if hasattr(simulation, "_body_model") and simulation._body_model:
                body_model_settings = {
                    "bodies": simulation._body_model.list_of_bodies(),
                    "global_frame_origin": simulation.global_frame_origin,
                    "global_frame_orientation": simulation.global_frame_orientation,
                }

        # Get simulation-level termination conditions
        try:
            termination_settings = json.loads(simulation.dump_termination_conditions())
        except (AttributeError, json.JSONDecodeError):
            # Fallback if dump method fails
            termination_settings = {}

        integrator_type = (
            type(simulation._integrator).__name__
            if hasattr(simulation, "_integrator") and simulation._integrator
            else None
        )

        # Check for existing simulation
        existing_sim = (
            session.query(Simulation)
            .filter(
                Simulation.simulation_type == simulation_type,
                Simulation.global_frame_origin == simulation.global_frame_origin,
                Simulation.global_frame_orientation
                == simulation.global_frame_orientation,
                Simulation.integrator_type == integrator_type,
                cast(Simulation.integrator_settings, JSONB) == integrator_settings,
                cast(Simulation.body_model_settings, JSONB) == body_model_settings,
                cast(Simulation.termination_settings, JSONB) == termination_settings,
            )
            .first()
        )

        if existing_sim:
            session.expunge(existing_sim)
            return existing_sim

        sim_record = Simulation(
            simulation_type=simulation_type,
            global_frame_origin=simulation.global_frame_origin,
            global_frame_orientation=simulation.global_frame_orientation,
            integrator_type=integrator_type,
            integrator_settings=integrator_settings,
            body_model_settings=body_model_settings,
            termination_settings=termination_settings,
        )

        session.add(sim_record)
        session.commit()
        session.refresh(sim_record)

        # Detach the object from the session so it can be used after the session closes
        session.expunge(sim_record)
        return sim_record

    if session:
        return _save_sim(session)
    else:
        with get_session_context() as session:
            return _save_sim(session)


def save_spacecraft(
    spacecraft, simulation_id: int, session: Optional[Session] = None
) -> Spacecraft:
    """
    Save a spacecraft configuration to the database.
    If an identical spacecraft for the same simulation exists, return that one.

    Args:
        spacecraft: The spacecraft instance
        simulation_id: ID of the associated simulation
        session: Optional database session (will create one if not provided)

    Returns:
        The created or existing Spacecraft database record
    """

    def _save_craft(session: Session) -> Spacecraft:
        # Serialize initial state
        initial_state_data = serialize_numpy_array(spacecraft._initial_state)

        # Use new dump methods for better serialization
        try:
            acceleration_settings = json.loads(
                spacecraft._dump_acceleration_settings()
            )
        except (AttributeError, json.JSONDecodeError):
            # Fallback to old method if dump method fails
            acceleration_settings = {}
            try:
                acc_settings = spacecraft._get_acceleration_settings()
                acceleration_settings = {
                    "type": "acceleration_settings",
                    "settings": str(acc_settings),
                }
            except Exception as e:
                acceleration_settings = {"error": str(e)}

        try:
            termination_settings = json.loads(spacecraft.dump_termination_settings())
        except (AttributeError, json.JSONDecodeError):
            # Fallback if dump method fails
            termination_settings = {
                "type": "termination_settings",
                "description": "Termination conditions",
            }

        # Store propagator settings as strings for now (no dump method available yet)
        propagator_settings = {
            "type": "propagator_settings",
            "description": "Propagator configuration",
        }

        name = spacecraft.get_name()
        spacecraft_type = type(spacecraft).__name__

        # Check for existing spacecraft
        existing_craft = (
            session.query(Spacecraft)
            .filter(
                Spacecraft.simulation_id == simulation_id,
                Spacecraft.name == name,
                Spacecraft.spacecraft_type == spacecraft_type,
                cast(Spacecraft.initial_state, JSONB) == initial_state_data,
                cast(Spacecraft.acceleration_settings, JSONB)
                == acceleration_settings,
                cast(Spacecraft.propagator_settings, JSONB) == propagator_settings,
                cast(Spacecraft.termination_settings, JSONB) == termination_settings,
            )
            .first()
        )

        if existing_craft:
            session.expunge(existing_craft)
            return existing_craft

        craft_record = Spacecraft(
            simulation_id=simulation_id,
            name=name,
            spacecraft_type=spacecraft_type,
            initial_state=initial_state_data,
            acceleration_settings=acceleration_settings,
            propagator_settings=propagator_settings,
            termination_settings=termination_settings,
            additional_properties={},
        )

        session.add(craft_record)
        session.commit()
        session.refresh(craft_record)

        # Detach the object from the session so it can be used after the session closes
        session.expunge(craft_record)
        return craft_record

    if session:
        return _save_craft(session)
    else:
        with get_session_context() as session:
            return _save_craft(session)


def save_trajectory(
    simulation_id: int,
    spacecraft_id: int,
    dynamics_simulator=None,
    trajectory_metadata: Optional[Dict[str, Any]] = None,
    session: Optional[Session] = None,
) -> Trajectory:
    """
    Create a new trajectory record for tracking simulation execution.

    :param simulation_id: ID of the associated simulation
    :param spacecraft_id: ID of the associated spacecraft
    :param dynamics_simulator: dynamics simulator object to serialize
    :param data_size: Optional size of the trajectory data (number of steps)
    :param start_time: Optional start time of the trajectory
    :param end_time: Optional end time of the trajectory
    :param trajectory_metadata: Optional metadata for the trajectory
    :param session: Optional database session (will create one if not provided)
    :return: The created Trajectory database record

    """

    def _create_trajectory(session: Session) -> Trajectory:
        # Serialize dynamics simulator if provided
        if dynamics_simulator is not None:
            dynamics_simulator_dict = serialize_dynamics_simulator(dynamics_simulator)
            termination_reason = dynamics_simulator_dict["termination_details"][
                "termination_reason"
            ]
            if isinstance(termination_reason, str) and termination_reason.startswith(
                "PropagationTerminationReason."
            ):
                termination_reason = termination_reason.replace(
                    "PropagationTerminationReason.", ""
                )

            trajectory_record = Trajectory(
                simulation_id=simulation_id,
                spacecraft_id=spacecraft_id,
                dynamics_simulator=json.dumps(
                    serialize_dynamics_simulator(dynamics_simulator)
                ),
                trajectory_metadata=trajectory_metadata or {},
                status="created",
                start_time=dynamics_simulator_dict["initial_and_final_times"][0],
                end_time=dynamics_simulator_dict["initial_and_final_times"][1],
                data_size=len(dynamics_simulator_dict["state_history"]),
                number_of_function_evaluations=dynamics_simulator_dict[
                    "total_number_of_function_evaluations"
                ],
                total_cpu_time=dynamics_simulator_dict["total_computation_time"],
                termination_reason=termination_reason,
                initial_state=(dynamics_simulator_dict["state_history"]["0.0"]),
            )
        else:
            trajectory_record = Trajectory(
                simulation_id=simulation_id,
                spacecraft_id=spacecraft_id,
                data_size=None,
                dynamics_simulator=None,
                trajectory_metadata=trajectory_metadata or {},
                status="created",
                start_time=None,
                end_time=None,
                number_of_function_evaluations=None,
                total_cpu_time=None,
            )

        session.add(trajectory_record)
        session.commit()
        session.refresh(trajectory_record)

        # Detach the object from the session so it can be used after the session closes
        session.expunge(trajectory_record)
        return trajectory_record

    if session:
        return _create_trajectory(session)
    else:
        with get_session_context() as session:
            return _create_trajectory(session)


def update_trajectory_status(
    trajectory_id: int,
    status: str,
    started_at: Optional[datetime] = None,
    completed_at: Optional[datetime] = None,
    error_message: Optional[str] = None,
    session: Optional[Session] = None,
    dynamics_simulator=None,
) -> Trajectory:
    """
    Update trajectory status and execution timestamps.

    Args:
        trajectory_id: ID of the trajectory to update
        status: New status ('running', 'completed', 'failed')
        started_at: Optional start timestamp
        completed_at: Optional completion timestamp
        error_message: Optional error message if status is 'failed'
        data_size: Optional number of data points/trajectory steps
        session: Optional database session

    Returns:
        The updated Trajectory record
    """

    def _update_status(session: Session) -> Trajectory:
        trajectory_record = (
            session.query(Trajectory).filter(Trajectory.id == trajectory_id).first()
        )
        if not trajectory_record:
            raise ValueError(f"Trajectory with ID {trajectory_id} not found")

        trajectory_record.status = status
        if started_at:
            trajectory_record.started_at = started_at
        if completed_at:
            trajectory_record.completed_at = completed_at
        if error_message:
            trajectory_record.error_message = error_message
        if dynamics_simulator is not None:
            dynamics_simulator_dict = serialize_dynamics_simulator(dynamics_simulator)
            termination_reason = dynamics_simulator_dict["termination_details"][
                "termination_reason"
            ]

            trajectory_record.start_time = dynamics_simulator_dict[
                "initial_and_final_times"
            ][0]
            trajectory_record.end_time = dynamics_simulator_dict[
                "initial_and_final_times"
            ][1]
            trajectory_record.dynamics_simulator = json.dumps(
                serialize_dynamics_simulator(dynamics_simulator)
            )
            trajectory_record.data_size = len(dynamics_simulator_dict["state_history"])
            trajectory_record.number_of_function_evaluations = (
                dynamics_simulator_dict["total_number_of_function_evaluations"]
            )
            trajectory_record.total_cpu_time = dynamics_simulator_dict[
                "total_computation_time"
            ]
            trajectory_record.initial_state = dynamics_simulator_dict["state_history"][
                "0.0"
            ]
            # Remove "PropagationTerminationReason." prefix if present
            if isinstance(termination_reason, str) and termination_reason.startswith(
                "PropagationTerminationReason."
            ):
                termination_reason_clean = termination_reason.replace(
                    "PropagationTerminationReason.", ""
                )
            else:
                termination_reason_clean = termination_reason
            trajectory_record.termination_reason = termination_reason_clean

        session.commit()
        session.refresh(trajectory_record)

        # Detach the object from the session so it can be used after the session closes
        session.expunge(trajectory_record)
        return trajectory_record

    if session:
        return _update_status(session)
    else:
        with get_session_context() as session:
            return _update_status(session)


def get_simulation(
    simulation_id: int, session: Optional[Session] = None
) -> Optional[Simulation]:
    """Get a simulation by ID."""

    def _get_sim(session: Session) -> Optional[Simulation]:
        sim_record = (
            session.query(Simulation).filter(Simulation.id == simulation_id).first()
        )
        if sim_record:
            # Detach the object from the session so it can be used after the session closes
            session.expunge(sim_record)
        return sim_record

    if session:
        return _get_sim(session)
    else:
        with get_session_context() as session:
            return _get_sim(session)


def get_spacecraft_by_simulation(
    simulation_id: int, session: Optional[Session] = None
) -> List[Spacecraft]:
    """Get all spacecraft for a simulation."""

    def _get_craft(session: Session) -> List[Spacecraft]:
        spacecraft_list = (
            session.query(Spacecraft)
            .filter(Spacecraft.simulation_id == simulation_id)
            .all()
        )
        # Detach all objects from the session so they can be used after the session closes
        for spacecraft in spacecraft_list:
            session.expunge(spacecraft)
        return spacecraft_list

    if session:
        return _get_craft(session)
    else:
        with get_session_context() as session:
            return _get_craft(session)


def get_simulation_dump_data(simulation) -> Dict[str, Any]:
    """
    Get comprehensive dump data from a simulation using the new dump methods.

    Args:
        simulation: The simulator instance

    Returns:
        Dictionary containing all dumped simulation data
    """
    dump_data = {}

    # Use the new dump methods with fallbacks
    try:
        dump_data["body_model"] = json.loads(simulation._dump_body_model())
    except (AttributeError, json.JSONDecodeError) as e:
        dump_data["body_model_error"] = str(e)

    try:
        dump_data["integrator_settings"] = json.loads(
            simulation._dump_integrator_settings()
        )
    except (AttributeError, json.JSONDecodeError) as e:
        dump_data["integrator_settings_error"] = str(e)

    try:
        dump_data["termination_conditions"] = json.loads(
            simulation.dump_termination_conditions()
        )
    except (AttributeError, json.JSONDecodeError) as e:
        dump_data["termination_conditions_error"] = str(e)

    return dump_data


def get_spacecraft_dump_data(spacecraft) -> Dict[str, Any]:
    """
    Get comprehensive dump data from a spacecraft using the new dump methods.

    Args:
        spacecraft: The spacecraft instance

    Returns:
        Dictionary containing all dumped spacecraft data
    """
    dump_data = {}

    # Use the new dump methods with fallbacks
    try:
        dump_data["acceleration_settings"] = json.loads(
            spacecraft._dump_acceleration_settings()
        )
    except (AttributeError, json.JSONDecodeError) as e:
        dump_data["acceleration_settings_error"] = str(e)

    try:
        dump_data["termination_settings"] = json.loads(
            spacecraft.dump_termination_settings()
        )
    except (AttributeError, json.JSONDecodeError) as e:
        dump_data["termination_settings_error"] = str(e)

    # Add basic spacecraft info
    dump_data["name"] = spacecraft.get_name()
    dump_data["type"] = type(spacecraft).__name__
    dump_data["initial_state"] = serialize_numpy_array(spacecraft._initial_state)

    return dump_data


def get_trajectory(
    trajectory_id: int, session: Optional[Session] = None
) -> Optional[Trajectory]:
    """Get a trajectory by ID."""

    def _get_trajectory(session: Session) -> Optional[Trajectory]:
        trajectory_record = (
            session.query(Trajectory).filter(Trajectory.id == trajectory_id).first()
        )
        if trajectory_record:
            # Detach the object from the session so it can be used after the session closes
            session.expunge(trajectory_record)
        return trajectory_record

    if session:
        return _get_trajectory(session)
    else:
        with get_session_context() as session:
            return _get_trajectory(session)


def get_trajectories_by_simulation(
    simulation_id: int, session: Optional[Session] = None
) -> List[Trajectory]:
    """Get all trajectories for a simulation."""

    def _get_trajectories(session: Session) -> List[Trajectory]:
        trajectory_list = (
            session.query(Trajectory)
            .filter(Trajectory.simulation_id == simulation_id)
            .all()
        )
        # Detach all objects from the session so they can be used after the session closes
        for trajectory in trajectory_list:
            session.expunge(trajectory)
        return trajectory_list

    if session:
        return _get_trajectories(session)
    else:
        with get_session_context() as session:
            return _get_trajectories(session)


def get_trajectories_by_spacecraft(
    spacecraft_id: int, session: Optional[Session] = None
) -> List[Trajectory]:
    """Get all trajectories for a spacecraft."""

    def _get_trajectories(session: Session) -> List[Trajectory]:
        trajectory_list = (
            session.query(Trajectory)
            .filter(Trajectory.spacecraft_id == spacecraft_id)
            .all()
        )
        # Detach all objects from the session so they can be used after the session closes
        for trajectory in trajectory_list:
            session.expunge(trajectory)
        return trajectory_list

    if session:
        return _get_trajectories(session)
    else:
        with get_session_context() as session:
            return _get_trajectories(session)


def get_simulation_status(
    simulation_id: int, session: Optional[Session] = None
) -> Optional[str]:
    """
    Get the current status of a simulation by checking its most recent trajectory.

    Args:
        simulation_id: ID of the simulation
        session: Optional database session

    Returns:
        Current status string or None if no trajectory exists
    """

    def _get_status(session: Session) -> Optional[str]:
        # Get the most recent trajectory for this simulation
        trajectory = (
            session.query(Trajectory)
            .filter(Trajectory.simulation_id == simulation_id)
            # .filter(Trajectory.individual_id.is_(None))  # Non-evolutionary trajectory
            .order_by(Trajectory.created_at.desc())
            .first()
        )
        return trajectory.status if trajectory else None

    if session:
        return _get_status(session)
    else:
        with get_session_context() as session:
            return _get_status(session)


def deserialize_dynamics_simulator(dynamics_simulator_json: str) -> dict:
    """
    Deserialize the JSON string stored in the dynamics_simulator column
    back into a Python dictionary, converting lists to numpy arrays where applicable.
    """
    if not dynamics_simulator_json:
        return None
    import numpy as np

    data = json.loads(dynamics_simulator_json)

    # Keys whose values should be dicts of arrays (e.g. state_history, dependent_variable_history)
    dict_of_arrays_keys = [
        "state_history",
        "dependent_variable_history",
        "cumulative_computation_time_history",
        "cumulative_number_of_function_evaluations",
    ]
    # Keys whose values should be arrays
    array_keys = ["initial_and_final_times"]

    def convert(obj, parent_key=None):
        if isinstance(obj, dict):
            new_obj = {}
            for k, v in obj.items():
                # If this is a dict of arrays, convert each value to np.array
                if parent_key in dict_of_arrays_keys:
                    new_obj[k] = np.array(v)
                else:
                    new_obj[k] = convert(v, k)
            return new_obj
        elif isinstance(obj, list):
            if parent_key in array_keys:
                return np.array(obj)
            # If all elements are numbers, treat as array
            if obj and all(isinstance(x, (int, float)) for x in obj):
                return np.array(obj)
            return [convert(x) for x in obj]
        else:
            return obj

    return convert(data)
