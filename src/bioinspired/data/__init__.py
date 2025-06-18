"""
Data access layer for managing trajectories, individuals, and experiments.
"""

import os
import numpy as np
from typing import List, Dict, Any, Tuple
from .models import Environment, Algorithm, Individual, Trajectory, Annotation
from .database import get_session


class TrajectoryManager:
    """Manages saving and loading of trajectory data."""
    
    def __init__(self, base_path: str = "data/trajectories"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_trajectory(self, trajectory_data: np.ndarray, 
                       individual_id: int, 
                       format: str = "npz") -> str:
        """Save trajectory data to file and return the file path."""
        filename = f"trajectory_{individual_id}.{format}"
        file_path = os.path.join(self.base_path, filename)
        
        if format == "npz":
            np.savez_compressed(file_path, trajectory=trajectory_data)
        elif format == "npy":
            np.save(file_path, trajectory_data)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Normalize path separators for database storage
        return file_path.replace("\\", "/")
    
    def load_trajectory(self, file_path: str) -> np.ndarray:
        """Load trajectory data from file."""
        # Normalize path for current OS
        normalized_path = file_path.replace("/", os.sep)
        
        if file_path.endswith(".npz"):
            data = np.load(normalized_path)
            return data["trajectory"]
        elif file_path.endswith(".npy"):
            return np.load(normalized_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


class DatabaseManager:
    """High-level interface for database operations."""
    
    def __init__(self):
        self.trajectory_manager = TrajectoryManager()
    
    def create_environment(self, name: str, description: str = None, 
                          parameters: Dict[str, Any] = None) -> Environment:
        """Create a new environment."""
        session = get_session()
        try:
            env = Environment(
                name=name,
                description=description,
                parameters=parameters or {}
            )
            session.add(env)
            session.commit()
            session.refresh(env)
            return env
        finally:
            session.close()
    
    def create_algorithm(self, environment_id: int, population_id: str, 
                        seed: int = None, hyperparameters: Dict[str, Any] = None) -> Algorithm:
        """Create a new algorithm run."""
        session = get_session()
        try:
            algorithm = Algorithm(
                environment_id=environment_id,
                population_id=population_id,
                seed=seed,
                hyperparameters=hyperparameters or {}
            )
            session.add(algorithm)
            session.commit()
            session.refresh(algorithm)
            return algorithm
        finally:
            session.close()
    
    def create_individual(self, algorithm_id: int, generation: int, 
                         fitness: float = None, species: str = None,
                         parameters: Dict[str, Any] = None) -> Individual:
        """Create a new individual."""
        session = get_session()
        try:
            individual = Individual(
                algorithm_id=algorithm_id,
                generation=generation,
                fitness=fitness,
                species=species,
                parameters=parameters or {}
            )
            session.add(individual)
            session.commit()
            session.refresh(individual)
            return individual
        finally:
            session.close()
    
    def save_trajectory_data(self, individual_id: int, trajectory_data: np.ndarray,
                           steps: int = None, format: str = "npz",
                           metadata: Dict[str, Any] = None) -> Trajectory:
        """Save trajectory data and create database record."""
        # Save the actual trajectory data to file
        file_path = self.trajectory_manager.save_trajectory(
            trajectory_data, individual_id, format
        )
        
        # Create database record
        session = get_session()
        try:
            trajectory = Trajectory(
                individual_id=individual_id,
                file_path=file_path,
                steps=steps or len(trajectory_data),
                format=format,
                trajectory_metadata=metadata or {}
            )
            session.add(trajectory)
            session.commit()
            session.refresh(trajectory)
            return trajectory
        finally:
            session.close()
    
    def load_trajectory_data(self, trajectory_id: int) -> Tuple[Trajectory, np.ndarray]:
        """Load trajectory data and metadata."""
        session = get_session()
        try:
            trajectory = session.query(Trajectory).filter_by(id=trajectory_id).first()
            if not trajectory:
                raise ValueError(f"Trajectory {trajectory_id} not found")
            
            # Load the actual trajectory data
            data = self.trajectory_manager.load_trajectory(trajectory.file_path)
            return trajectory, data
        finally:
            session.close()
    
    def get_best_individuals(self, algorithm_id: int, limit: int = 10) -> List[Individual]:
        """Get the best performing individuals from an algorithm run."""
        session = get_session()
        try:
            return session.query(Individual).filter_by(
                algorithm_id=algorithm_id
            ).order_by(Individual.fitness.desc()).limit(limit).all()
        finally:
            session.close()
    
    def get_generation_stats(self, algorithm_id: int) -> List[Dict[str, Any]]:
        """Get fitness statistics by generation."""
        session = get_session()
        try:
            from sqlalchemy import func
            
            results = session.query(
                Individual.generation,
                func.count(Individual.id).label('count'),
                func.avg(Individual.fitness).label('avg_fitness'),
                func.max(Individual.fitness).label('max_fitness'),
                func.min(Individual.fitness).label('min_fitness')
            ).filter_by(algorithm_id=algorithm_id).group_by(
                Individual.generation
            ).order_by(Individual.generation).all()
            
            return [
                {
                    'generation': r.generation,
                    'count': r.count,
                    'avg_fitness': float(r.avg_fitness) if r.avg_fitness else None,
                    'max_fitness': float(r.max_fitness) if r.max_fitness else None,
                    'min_fitness': float(r.min_fitness) if r.min_fitness else None
                }
                for r in results
            ]
        finally:
            session.close()
    
    def add_annotation(self, individual_id: int, annotation_type: str,
                      payload: Dict[str, Any] = None) -> Annotation:
        """Add an annotation to an individual."""
        session = get_session()
        try:
            annotation = Annotation(
                individual_id=individual_id,
                type=annotation_type,
                payload=payload or {}
            )
            session.add(annotation)
            session.commit()
            session.refresh(annotation)
            return annotation
        finally:
            session.close()
