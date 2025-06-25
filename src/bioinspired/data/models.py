"""
Database models for the BioInspired evolutionary docking project.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import zoneinfo


def get_amsterdam_now():
    """Get current Amsterdam datetime."""
    amsterdam_tz = zoneinfo.ZoneInfo("Europe/Amsterdam")
    return datetime.now(amsterdam_tz)


Base = declarative_base()

# Evolutionary algorithm classes - commented out for now
# These will be uncommented when implementing the evolutionary features

# class Algorithm(Base):
#     """
#     Represents an algorithm run within a specific environment.
#     Tracks hyperparameters and settings for reproducibility.
#     """
#     __tablename__ = "algorithms"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     # ... other fields ...

# class Individual(Base):
#     """
#     Represents a candidate solution (individual) in the evolutionary algorithm.
#     Each individual has a fitness score and belongs to a specific generation.
#     """

#     __tablename__ = "individuals"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     algorithm_id = Column(Integer, ForeignKey("algorithms.id"), nullable=False)
#     generation = Column(Integer, nullable=False)
#     species = Column(String(255))  # For tracking species/clusters
#     fitness = Column(Float)
#     created_at = Column(DateTime, default=get_amsterdam_now)
#     parameters = Column(JSON)  # Individual-specific parameters

#     # Relationships
#     algorithm = relationship("Algorithm", back_populates="individuals")
#     trajectories = relationship("Trajectory", back_populates="individual")
#     annotations = relationship("Annotation", back_populates="individual")

#     def __repr__(self):
#         return f"<Individual(id={self.id}, generation={self.generation}, fitness={self.fitness})>"


class Trajectory(Base):
    """
    Represents trajectory data and simulation results for a simulation run.
    Large numerical data is stored in files, with metadata here.
    Includes status tracking fields and the dynamics simulator object as a blob.
    Merged functionality from SimulationResult for comprehensive trajectory storage.
    """

    __tablename__ = "trajectories"
    id = Column(Integer, primary_key=True, autoincrement=True)

    # For evolutionary algorithm integration (optional for now)
    # individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=True)  # Commented out until evolutionary features are enabled

    # For direct simulation tracking
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    spacecraft_id = Column(Integer, ForeignKey("spacecraft.id"), nullable=False)

    # Trajectory/Result data details
    data_size = Column(Integer)  # Number of data points/trajectory steps
    start_time = Column(Float)  # Simulation start time
    end_time = Column(Float)  # Simulation end time

    # Simulator object storage
    dynamics_simulator = Column(JSON, nullable=True)  # Store simulator as a dict/JSON
    initial_state = Column(JSON, nullable=True)  # Initial state of the spacecraft
    number_of_function_evaluations = Column(
        Integer, nullable=True
    )  # Number of function evaluations
    total_cpu_time = Column(Float, nullable=True)  # Total CPU time used
    termination_reason = Column(String(255), nullable=True)
    # Metadata
    trajectory_metadata = Column(JSON)  # Additional trajectory metadata

    # Status tracking fields (moved from Simulation model)
    status = Column(
        String(50), default="created"
    )  # created, running, completed, failed
    created_at = Column(DateTime, default=get_amsterdam_now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    # individual = relationship("Individual", back_populates="trajectories")  # Uncomment when evolutionary features are enabled
    simulation = relationship("Simulation", back_populates="trajectories")
    spacecraft = relationship("Spacecraft", back_populates="trajectories")

    def __repr__(self):
        return f"<Trajectory(id={self.id}, simulation_id={self.simulation_id}, spacecraft_id={self.spacecraft_id}, status='{self.status}', size={self.data_size})>"


# class Annotation(Base):
#     """
#     Represents annotations or analysis results for individuals.
#     Flexible schema for adding comments, clusters, or other derived data.
#     """

#     __tablename__ = "annotations"

#     id = Column(Integer, primary_key=True, autoincrement=True)
#     individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False)
#     type = Column(String(100), nullable=False)  # e.g., "cluster", "remark", "selected"
#     payload = Column(JSON)  # Arbitrary additional information
#     created_at = Column(DateTime, default=get_amsterdam_now)

#     # Relationships
#     individual = relationship("Individual", back_populates="annotations")

#     def __repr__(self):
#         return f"<Annotation(id={self.id}, type='{self.type}')>"


class Simulation(Base):
    """
    Represents a simulation configuration and execution.
    Stores simulation parameters, settings, and execution metadata.
    Status tracking is now handled in the Trajectory model.
    Results are also stored in Trajectory model (merged from SimulationResult).
    """

    __tablename__ = "simulations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_type = Column(
        String(100), nullable=False
    )  # e.g., "EmptyUniverseSimulator"
    global_frame_origin = Column(String(50), default="SSB")
    global_frame_orientation = Column(String(50), default="ECLIPJ2000")
    integrator_type = Column(String(100))  # e.g., "runge_kutta_fixed_step"
    integrator_settings = Column(JSON)  # Store integrator configuration
    body_model_settings = Column(JSON)  # Store body model configuration
    termination_settings = Column(JSON)  # Store simulation-level termination conditions

    # Relationships
    spacecraft = relationship("Spacecraft", back_populates="simulation")
    trajectories = relationship("Trajectory", back_populates="simulation")

    def __repr__(self):
        return f"<Simulation(id={self.id}, type='{self.simulation_type}')>"


class Spacecraft(Base):
    """
    Represents a spacecraft configuration and properties.
    Stores spacecraft design parameters, initial conditions, and physical properties.
    """

    __tablename__ = "spacecraft"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(Integer, ForeignKey("simulations.id"), nullable=False)
    name = Column(String(255), nullable=False)
    spacecraft_type = Column(String(100), nullable=False)  # e.g., "SimpleCraft"
    initial_state = Column(
        JSON, nullable=False
    )  # [x, y, z, vx, vy, vz] in proper units
    acceleration_settings = Column(JSON)  # Store acceleration model configuration
    propagator_settings = Column(JSON)  # Store propagator configuration
    termination_settings = Column(JSON)  # Store spacecraft-level termination conditions
    created_at = Column(DateTime, default=get_amsterdam_now)
    additional_properties = Column(JSON)  # Any additional spacecraft properties

    # Relationships
    simulation = relationship("Simulation", back_populates="spacecraft")
    trajectories = relationship("Trajectory", back_populates="spacecraft")

    def __repr__(self):
        return f"<Spacecraft(id={self.id}, name='{self.name}', type='{self.spacecraft_type}')>"
