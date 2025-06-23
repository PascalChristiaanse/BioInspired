"""Data module for bioinspired package.
This module contains the database management and data handling
for the bioinspired package. It includes functions to initialize the database,
create tables, and manage data storage.
It is designed to work with PostgreSQL and uses SQLAlchemy for ORM.
"""

from .init_db import init_database as init_db
from .models import Simulation, Spacecraft, Trajectory
from .database import get_session, get_session_context, create_tables
from .services import (
    save_simulation,
    save_spacecraft,
    save_trajectory,
    update_trajectory_status,
    get_simulation,
    get_spacecraft_by_simulation,
    get_trajectory,
    get_trajectories_by_simulation,
    get_simulation_status,
)

__all__ = [
    "init_db",
    # Models
    "Trajectory",
    "Simulation",
    "Spacecraft",
    # Database functions
    "get_session",
    "get_session_context",
    "create_tables",
    # Services
    "save_simulation",
    "save_spacecraft",
    "save_trajectory",
    "update_trajectory_status",
    "get_simulation",
    "get_spacecraft_by_simulation",
    "get_trajectory",
    "get_trajectories_by_simulation",
    "get_simulation_status",
]
