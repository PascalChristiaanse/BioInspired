"""
Database models for the BioInspired evolutionary docking project.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Environment(Base):
    """
    Represents a simulated environment where evolutionary algorithms run.
    Different environments can have varying complexity levels.
    """
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON)  # Environment-specific settings
    
    # Relationships
    algorithms = relationship("Algorithm", back_populates="environment")

    def __repr__(self):
        return f"<Environment(id={self.id}, name='{self.name}')>"


class Algorithm(Base):
    """
    Represents an algorithm run within a specific environment.
    Tracks hyperparameters and settings for reproducibility.
    """
    __tablename__ = "algorithms"

    id = Column(Integer, primary_key=True, autoincrement=True)
    environment_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    population_id = Column(String(255))  # For distinguishing separate runs
    seed = Column(Integer)  # For reproducibility
    hyperparameters = Column(JSON)  # Algorithm settings
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default="running")  # running, completed, failed
    
    # Relationships
    environment = relationship("Environment", back_populates="algorithms")
    individuals = relationship("Individual", back_populates="algorithm")

    def __repr__(self):
        return f"<Algorithm(id={self.id}, population_id='{self.population_id}')>"


class Individual(Base):
    """
    Represents a candidate solution (individual) in the evolutionary algorithm.
    Each individual has a fitness score and belongs to a specific generation.
    """
    __tablename__ = "individuals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    algorithm_id = Column(Integer, ForeignKey("algorithms.id"), nullable=False)
    generation = Column(Integer, nullable=False)
    species = Column(String(255))  # For tracking species/clusters
    fitness = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    parameters = Column(JSON)  # Individual-specific parameters
    
    # Relationships
    algorithm = relationship("Algorithm", back_populates="individuals")
    trajectories = relationship("Trajectory", back_populates="individual")
    annotations = relationship("Annotation", back_populates="individual")

    def __repr__(self):
        return f"<Individual(id={self.id}, generation={self.generation}, fitness={self.fitness})>"


class Trajectory(Base):
    """
    Represents trajectory data for an individual.
    Large numerical data is stored in files, with metadata here.
    """
    __tablename__ = "trajectories"    
    id = Column(Integer, primary_key=True, autoincrement=True)
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path to trajectory data file
    steps = Column(Integer)  # Number of trajectory points
    format = Column(String(10), default="npz")  # File format (npz, hdf5, etc.)
    created_at = Column(DateTime, default=datetime.utcnow)
    trajectory_metadata = Column(JSON)  # Additional trajectory metadata
    
    # Relationships
    individual = relationship("Individual", back_populates="trajectories")

    def __repr__(self):
        return f"<Trajectory(id={self.id}, individual_id={self.individual_id}, steps={self.steps})>"


class Annotation(Base):
    """
    Represents annotations or analysis results for individuals.
    Flexible schema for adding comments, clusters, or other derived data.
    """
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    individual_id = Column(Integer, ForeignKey("individuals.id"), nullable=False)
    type = Column(String(100), nullable=False)  # e.g., "cluster", "remark", "selected"
    payload = Column(JSON)  # Arbitrary additional information
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    individual = relationship("Individual", back_populates="annotations")

    def __repr__(self):
        return f"<Annotation(id={self.id}, type='{self.type}')>"
