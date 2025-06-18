"""
Basic evolutionary algorithm implementation for docking optimization.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
import random


class Individual:
    """Represents an individual solution in the evolutionary algorithm."""
    
    def __init__(self, genes: np.ndarray, fitness: float = None):
        self.genes = genes
        self.fitness = fitness
        self.trajectory = None
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.3f})"


class EvolutionaryAlgorithm:
    """Basic evolutionary algorithm for docking optimization."""
    
    def __init__(self, 
                 population_size: int = 50,
                 gene_length: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 seed: int = None):
        """
        Initialize the evolutionary algorithm.
        
        Args:
            population_size: Number of individuals in the population
            gene_length: Length of the gene vector for each individual
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of best individuals to preserve
            seed: Random seed for reproducibility
        """
        self.population_size = population_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.fitness_history = []
    
    def initialize_population(self) -> List[Individual]:
        """Initialize a random population."""
        self.population = []
        for _ in range(self.population_size):
            genes = np.random.uniform(-1, 1, self.gene_length)
            individual = Individual(genes)
            self.population.append(individual)
        return self.population
    
    def evaluate_fitness(self, individual: Individual) -> float:
        """
        Evaluate the fitness of an individual.
        This is a placeholder - replace with your actual docking scoring function.
        """
        # Simple example: maximize sum of squares with some noise
        base_fitness = np.sum(individual.genes ** 2)
        noise = np.random.normal(0, 0.1)  # Add some noise
        fitness = base_fitness + noise
        
        # Generate a simple trajectory as an example
        trajectory = self._generate_trajectory(individual.genes)
        individual.trajectory = trajectory
        
        return fitness
    
    def _generate_trajectory(self, genes: np.ndarray) -> np.ndarray:
        """Generate a trajectory based on the individual's genes."""
        # Simple trajectory: spiral based on gene values
        steps = 100
        t = np.linspace(0, 4 * np.pi, steps)
        
        # Use genes to parameterize the trajectory
        radius = abs(genes[0]) + 0.1
        height_factor = genes[1] if len(genes) > 1 else 0.5
        frequency = abs(genes[2]) + 0.1 if len(genes) > 2 else 1.0
        
        x = radius * np.cos(frequency * t)
        y = radius * np.sin(frequency * t)
        z = height_factor * t
        
        trajectory = np.column_stack([x, y, z])
        return trajectory
    
    def selection(self, tournament_size: int = 3) -> Individual:
        """Tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Single-point crossover."""
        if random.random() > self.crossover_rate:
            return Individual(parent1.genes.copy()), Individual(parent2.genes.copy())
        
        crossover_point = random.randint(1, self.gene_length - 1)
        
        child1_genes = np.concatenate([
            parent1.genes[:crossover_point],
            parent2.genes[crossover_point:]
        ])
        
        child2_genes = np.concatenate([
            parent2.genes[:crossover_point],
            parent1.genes[crossover_point:]
        ])
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation."""
        genes = individual.genes.copy()
        
        for i in range(len(genes)):
            if random.random() < self.mutation_rate:
                genes[i] += np.random.normal(0, 0.1)
                genes[i] = np.clip(genes[i], -2, 2)  # Keep within bounds
        
        return Individual(genes)
    
    def evolve_generation(self):
        """Evolve one generation."""
        # Evaluate fitness for all individuals
        for individual in self.population:
            if individual.fitness is None:
                individual.fitness = self.evaluate_fitness(individual)
        
        # Sort by fitness (descending)
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Update best individual
        if self.best_individual is None or self.population[0].fitness > self.best_individual.fitness:
            self.best_individual = Individual(self.population[0].genes.copy())
            self.best_individual.fitness = self.population[0].fitness
            self.best_individual.trajectory = self.population[0].trajectory
        
        # Record statistics
        fitnesses = [ind.fitness for ind in self.population]
        self.fitness_history.append({
            'generation': self.generation,
            'max_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'min_fitness': min(fitnesses),
            'std_fitness': np.std(fitnesses)
        })
        
        # Create new population
        new_population = []
        
        # Elitism: keep best individuals
        elite_count = int(self.population_size * self.elitism_rate)
        new_population.extend([
            Individual(ind.genes.copy()) for ind in self.population[:elite_count]
        ])
        
        # Generate rest of population through selection and crossover
        while len(new_population) < self.population_size:
            parent1 = self.selection()
            parent2 = self.selection()
            
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        self.population = new_population[:self.population_size]
        self.generation += 1
    
    def run(self, num_generations: int) -> Dict[str, Any]:
        """Run the evolutionary algorithm for a specified number of generations."""
        if not self.population:
            self.initialize_population()
        
        for _ in range(num_generations):
            self.evolve_generation()
        
        return {
            'best_individual': self.best_individual,
            'fitness_history': self.fitness_history,
            'final_population': self.population,
            'generations': self.generation
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get algorithm hyperparameters for storage."""
        return {
            'population_size': self.population_size,
            'gene_length': self.gene_length,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'elitism_rate': self.elitism_rate
        }
    
    def calculate_trajectory_smoothness(self, trajectory: np.ndarray) -> float:
        """Calculate trajectory smoothness score (lower is smoother)."""
        if len(trajectory) < 3:
            return 0.0
            
        # Calculate acceleration (second derivative of position)
        velocities = np.diff(trajectory[:, :3], axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Calculate jerk (third derivative)
        jerks = np.diff(accelerations, axis=0)
        
        # Smoothness is inverse of average jerk magnitude
        jerk_magnitudes = np.linalg.norm(jerks, axis=1)
        avg_jerk = np.mean(jerk_magnitudes)
        
        # Return smoothness score (0-1, where 1 is perfectly smooth)
        return 1.0 / (1.0 + avg_jerk)
