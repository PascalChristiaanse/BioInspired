"""
PyGMO-specific database models for evolutionary optimization tracking.
These models are separate from the simulation models to allow independent development.
"""

from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    Text,
    DateTime,
    Float,
    ForeignKey,
    Boolean,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from datetime import datetime
from typing import Optional, List, Dict

import zoneinfo
import platform
import multiprocessing

from bioinspired.data.database import get_session_context

import enum


class STATUS(enum.Enum):
    INITIALIZED = "initialized"
    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def get_amsterdam_now():
    """Get current Amsterdam datetime."""
    return datetime.now(zoneinfo.ZoneInfo("Europe/Amsterdam"))


base = declarative_base()


class PyGMOBase(base):
    """
    Base class for all PyGMO-related database models.
    Provides common functionality and metadata.
    """

    __abstract__ = True  # This tells SQLAlchemy not to create a table for this class

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_to_db(self, session: Optional[Session] = None, commit: bool = True):
        """
        Save the current instance to the database using database.py pattern.

        Args:
            session: Optional SQLAlchemy session. If None, creates a new session.
            commit: Whether to commit the transaction (default: True).

        Returns:
            The saved instance (with any auto-generated fields populated).
        """

        def _save_instance(session: Session):
            session.add(self)
            if commit:
                session.commit()
                session.refresh(self)
            # Detach from session so it can be used after session closes
            session.expunge(self)
            return self

        if session is not None:
            return _save_instance(session)
        else:
            with get_session_context() as session:
                return _save_instance(session)

    def update_in_db(
        self, session: Optional[Session] = None, commit: bool = True, **kwargs
    ):
        """
        Update the current instance with new values using database.py pattern.

        Args:
            session: Optional SQLAlchemy session.
            commit: Whether to commit the transaction.
            **kwargs: Fields to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"'{self.__class__.__name__}' has no attribute '{key}'"
                )

        return self.save_to_db(session=session, commit=commit)

    def delete_from_db(self, session: Optional[Session] = None, commit: bool = True):
        """
        Delete the current instance from the database using database.py pattern.

        Args:
            session: Optional SQLAlchemy session.
            commit: Whether to commit the transaction.
        """

        def _delete_instance(session: Session):
            session.delete(self)
            if commit:
                session.commit()

        if session is not None:
            _delete_instance(session)
        else:
            with get_session_context() as session:
                _delete_instance(session)

    @classmethod
    def get_by_id(cls, record_id: int, session: Optional[Session] = None):
        """
        Get a record by ID using database.py pattern.

        Args:
            record_id: ID of the record to retrieve
            session: Optional SQLAlchemy session

        Returns:
            The record instance or None if not found
        """

        def _get_record(session: Session):
            record = session.get(cls, record_id)
            if record:
                session.expunge(record)
            return record

        if session is not None:
            return _get_record(session)
        else:
            with get_session_context() as session:
                return _get_record(session)

    @classmethod
    def create_tables(cls, engine=None):
        """Create all tables for PyGMO models."""
        from bioinspired.data.database import engine as default_engine

        target_engine = engine or default_engine
        base.metadata.create_all(target_engine)

    @classmethod
    def get_session_context_class(cls):
        """Class method version of get_session_context for class methods."""
        return get_session_context()

    def to_dict(self, include_relationships: bool = False):
        """
        Convert the instance to a dictionary.

        Args:
            include_relationships: Whether to include relationship fields.
        """
        result = {}

        # Include column attributes
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            else:
                result[column.name] = value

        # Optionally include relationships
        if include_relationships:
            for relationship_name in self.__mapper__.relationships.keys():
                relationship_value = getattr(self, relationship_name)
                if relationship_value is not None:
                    if hasattr(relationship_value, "__iter__") and not isinstance(
                        relationship_value, str
                    ):
                        # Collection relationship
                        result[relationship_name] = [
                            item.to_dict() if hasattr(item, "to_dict") else str(item)
                            for item in relationship_value
                        ]
                    else:
                        # Single relationship
                        result[relationship_name] = (
                            relationship_value.to_dict()
                            if hasattr(relationship_value, "to_dict")
                            else str(relationship_value)
                        )

        return result


class Archipelago(PyGMOBase):
    """
    Top-level table for tracking PyGMO archipelago optimization runs.
    Each archipelago represents one complete optimization experiment.
    """

    __tablename__ = "archipelago"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Basic identification
    name = Column(String(255), nullable=True)  # Optional user-defined name
    description = Column(Text, nullable=True)  # Optional description

    # Problem configuration
    problem_class = Column(
        String(255), nullable=False
    )  # e.g., "StopNeuronBasicProblem"
    problem_parameters = Column(JSON, nullable=True)  # Problem-specific configuration

    # Archipelago configuration
    num_islands = Column(Integer, nullable=False)
    population_per_island = Column(Integer, nullable=False)
    topology_type = Column(
        String(100), default="fully_connected"
    )  # "fully_connected", "ring", etc.
    topology_parameters = Column(JSON, nullable=True)  # Additional topology parameters

    # Evolution configuration
    generations_per_migration_event = Column(Integer, nullable=False)
    migration_events = Column(Integer, nullable=False)
    total_generations = Column(Integer, nullable=False)  # Calculated field

    # Execution environment
    seed = Column(Integer, nullable=False)
    num_cores_available = Column(
        Integer, nullable=True, default=multiprocessing.cpu_count()
    )
    num_cores_used = Column(Integer, nullable=True, default=multiprocessing.cpu_count())
    system_name = Column(
        String(100), nullable=True, default=platform.node()
    )  # Name of the machine running the experiment (Bingus, pc-pascal, etc.)

    # Status and timing
    status = Column(
        String(50), default=STATUS.INITIALIZED.value
    )  # "created", "running", "completed", "failed"
    created_at = Column(DateTime, default=get_amsterdam_now)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    total_runtime_seconds = Column(Float, nullable=True)
    exception_message = Column(
        Text, nullable=True
    )  # For failed runs    # Results summary
    best_fitness = Column(Float, nullable=True)
    worst_fitness = Column(Float, nullable=True)
    average_fitness = Column(Float, nullable=True)
    total_evaluations = Column(BigInteger, nullable=True)

    # Migration tracking for cross-process communication
    current_migration_event = Column(
        Integer, default=0
    )  # Current migration cycle number
    # Relationships
    islands = relationship(
        "Island", back_populates="archipelago", cascade="all, delete-orphan"
    )
    individuals = relationship(
        "Individual", back_populates="archipelago", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Archipelago(id={self.id}, name='{self.name}', status='{self.status}', islands={self.num_islands})>"

    @classmethod
    def create_new(
        cls,
        name: str,
        problem_class: str,
        num_islands: int,
        population_per_island: int,
        generations_per_migration: int,
        migration_events: int,
        seed: int,
        description: str = None,
        problem_parameters: Dict = None,
        algorithm_type: str = "sga",
        **kwargs,
    ) -> "Archipelago":
        """
        Create a new archipelago record with configuration.
        Similar to save_simulation pattern in services.py.
        """

        def _create_archipelago(session: Session) -> "Archipelago":
            import socket

            archipelago = cls(
                name=name,
                description=description,
                problem_class=problem_class,
                problem_parameters=problem_parameters or {},
                num_islands=num_islands,
                population_per_island=population_per_island,
                topology_type="fully_connected",
                topology_parameters={},
                generations_per_migration_event=generations_per_migration,
                migration_events=migration_events,
                total_generations=generations_per_migration * migration_events,
                seed=seed,
                num_cores_available=kwargs.get("num_cores_available"),
                num_cores_used=kwargs.get("num_cores_used"),
                system_name=socket.gethostname(),
                status=STATUS.INITIALIZED.value,
                current_migration_event=0,
            )

            session.add(archipelago)
            session.commit()
            session.refresh(archipelago)
            session.expunge(archipelago)
            return archipelago

        with cls.get_session_context_class() as session:
            return _create_archipelago(session)

class Island(PyGMOBase):
    """
    Represents an individual island in the archipelago.
    Tracks island-specific configuration and performance.
    """

    __tablename__ = "island"

    id = Column(Integer, primary_key=True, autoincrement=True)
    archipelago_id = Column(Integer, ForeignKey("archipelago.id"), nullable=False)

    # Island identification
    island_id = Column(
        Integer, nullable=False
    )  # 0, 1, 2, etc. (position in archipelago)

    # Island configuration
    initial_population_size = Column(Integer, nullable=False)
    algorithm_type = Column(
        String(100), nullable=False
    )  # May differ per island in future
    algorithm_parameters = Column(JSON, nullable=True)

    # Island performance tracking
    generations_completed = Column(Integer, default=0)
    best_fitness = Column(Float, nullable=True)
    worst_fitness = Column(Float, nullable=True)
    # Status
    status = Column(
        String(50),
        default=STATUS.INITIALIZED.value,
    )  # "created", "evolving", "completed", "failed"
    error_message = Column(Text, nullable=True)  # Relationships
    archipelago = relationship("Archipelago", back_populates="islands")
    individuals = relationship(
        "Individual", back_populates="island", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Island(id={self.id}, archipelago_id={self.archipelago_id}, index={self.island_id}, status='{self.status}')>"


class Individual(PyGMOBase):
    """
    Represents an individual (solution candidate) in the evolutionary algorithm.
    Can be a regular individual or marked as a champion.
    """

    __tablename__ = "individual"

    id = Column(Integer, primary_key=True, autoincrement=True)
    archipelago_id = Column(Integer, ForeignKey("archipelago.id"), nullable=False)
    island_id = Column(Integer, ForeignKey("island.id"), nullable=False)

    # Individual identification
    individual_ID = Column(Integer, nullable=False)  # Individual ID within population
    generation_number = Column(
        Integer, nullable=False
    )  # Which generation this individual belongs to
    migration_event = Column(
        Integer, nullable=False
    )  # Migration event number (0, 1, 2, etc.)
    total_generation = Column(
        Integer, nullable=False
    )  # Total generation number across all migrations, product of migration event and generation number

    # Individual data
    chromosome = Column(JSON, nullable=False)  # Neural network weights as array
    chromosome_size = Column(Integer, nullable=False)  # Number of parameters

    # Fitness data (simplified - single fitness value)
    fitness = Column(Float, nullable=False)  # Primary fitness for evolution
    is_champion = Column(Boolean, default=False)  # True if this was the best individual

    # Metadata
    created_at = Column(DateTime, default=get_amsterdam_now)
    additional_metrics = Column(JSON, nullable=True)  # Individual-specific metrics

    # Relationships
    archipelago = relationship("Archipelago", back_populates="individuals")
    island = relationship("Island", back_populates="individuals")

    def __repr__(self):
        champion_str = " (CHAMPION)" if self.is_champion else ""
        fitness_str = f"{self.fitness:.6f}" if self.fitness else "N/A"
        return f"<Individual(id={self.id}, island={self.island_id}, gen={self.generation_number}, fitness={fitness_str}{champion_str})>"
