"""
Visualization tools for plotting evolutionary algorithm results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
import seaborn as sns

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")


class EvolutionPlotter:
    """Visualization tools for evolutionary algorithm results."""
    
    def __init__(self, figsize: tuple = (12, 8)):
        self.figsize = figsize
    
    def plot_fitness_evolution(self, fitness_history: List[Dict[str, Any]], 
                             save_path: str = None, show: bool = True):
        """Plot fitness evolution over generations."""
        generations = [entry['generation'] for entry in fitness_history]
        max_fitness = [entry['max_fitness'] for entry in fitness_history]
        avg_fitness = [entry['avg_fitness'] for entry in fitness_history]
        min_fitness = [entry['min_fitness'] for entry in fitness_history]
        std_fitness = [entry['std_fitness'] for entry in fitness_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Plot fitness trends
        ax1.plot(generations, max_fitness, 'r-', label='Max Fitness', linewidth=2)
        ax1.plot(generations, avg_fitness, 'b-', label='Avg Fitness', linewidth=2)
        ax1.plot(generations, min_fitness, 'g-', label='Min Fitness', linewidth=2)
        ax1.fill_between(generations, 
                        np.array(avg_fitness) - np.array(std_fitness),
                        np.array(avg_fitness) + np.array(std_fitness),
                        alpha=0.3, color='blue', label='Std Dev')
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Evolution Over Generations')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot fitness diversity
        ax2.plot(generations, std_fitness, 'purple', linewidth=2)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Standard Deviation')
        ax2.set_title('Population Diversity Over Generations')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_trajectory_3d(self, trajectory: np.ndarray, 
                          title: str = "Trajectory Visualization",
                          save_path: str = None, show: bool = True):
        """Plot a 3D trajectory."""
        if trajectory.shape[1] < 3:
            raise ValueError("Trajectory must have at least 3 dimensions for 3D plotting")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create color gradient based on time
        colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory)))
        
        # Plot trajectory
        ax.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                  c=colors, s=20, alpha=0.7)
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
               'k-', alpha=0.3, linewidth=1)
        
        # Mark start and end points
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='red', s=100, marker='s', label='End')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_population_fitness_distribution(self, population_fitness: List[float],
                                           generation: int = None,
                                           save_path: str = None, show: bool = True):
        """Plot fitness distribution of a population."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        # Histogram
        ax1.hist(population_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(population_fitness), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(population_fitness):.3f}')
        ax1.axvline(np.median(population_fitness), color='green', linestyle='--', 
                   label=f'Median: {np.median(population_fitness):.3f}')
        ax1.set_xlabel('Fitness')
        ax1.set_ylabel('Count')
        title = 'Fitness Distribution'
        if generation is not None:
            title += f' (Generation {generation})'
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(population_fitness, vert=True)
        ax2.set_ylabel('Fitness')
        ax2.set_title('Fitness Box Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_species_analysis(self, species_data: Dict[str, List[float]],
                            save_path: str = None, show: bool = True):
        """Plot analysis of different species in the population."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        species_names = list(species_data.keys())
        species_fitnesses = list(species_data.values())
        
        # Box plot comparing species
        ax1.boxplot(species_fitnesses, labels=species_names)
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Fitness Distribution by Species')
        ax1.grid(True, alpha=0.3)
        
        # Average fitness by species
        avg_fitnesses = [np.mean(fitnesses) for fitnesses in species_fitnesses]
        ax2.bar(species_names, avg_fitnesses, color='lightcoral')
        ax2.set_xlabel('Species')
        ax2.set_ylabel('Average Fitness')
        ax2.set_title('Average Fitness by Species')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_convergence_analysis(self, fitness_history: List[Dict[str, Any]],
                                save_path: str = None, show: bool = True):
        """Analyze convergence patterns."""
        generations = [entry['generation'] for entry in fitness_history]
        max_fitness = [entry['max_fitness'] for entry in fitness_history]
        avg_fitness = [entry['avg_fitness'] for entry in fitness_history]
        
        # Calculate improvement rate
        improvement_rate = []
        for i in range(1, len(max_fitness)):
            rate = max_fitness[i] - max_fitness[i-1]
            improvement_rate.append(rate)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Fitness convergence
        ax1.plot(generations, max_fitness, 'r-', linewidth=2, label='Max Fitness')
        ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Avg Fitness')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Fitness')
        ax1.set_title('Convergence Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Improvement rate
        ax2.plot(generations[1:], improvement_rate, 'g-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Improvement Rate')
        ax2.set_title('Fitness Improvement Rate Over Time')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_multiple_trajectories(self, trajectories: List[np.ndarray], 
                                 labels: List[str] = None,
                                 title: str = "Multiple Trajectories",
                                 save_path: str = None, 
                                 show: bool = False) -> plt.Figure:
        """Plot multiple trajectories for comparison."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(trajectories)))
        
        for i, trajectory in enumerate(trajectories):
            label = labels[i] if labels and i < len(labels) else f"Trajectory {i+1}"
            color = colors[i]
            
            # Plot trajectory path
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   color=color, linewidth=2, alpha=0.8, label=label)
            
            # Mark start and end points
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                      c=color, s=80, marker='o', alpha=0.8)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      c=color, s=80, marker='s', alpha=0.8)
        
        # Add target point at origin
        ax.scatter(0, 0, 0, c='gold', s=200, marker='*', label='Target')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title(title)
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
