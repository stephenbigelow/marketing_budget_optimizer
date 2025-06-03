import numpy as np
from typing import List, Dict, Optional, Protocol
from abc import ABC, abstractmethod
from .curves import ResponseCurve

class OptimizationMethod(Protocol):
    """Protocol defining the interface for optimization methods."""
    
    def optimize(self, 
                curves: List[ResponseCurve],
                budget: float,
                min_budgets: List[float],
                max_budgets: List[float]) -> Dict[str, np.ndarray]:
        """
        Optimize budget allocation using the specific method.
        
        Args:
            curves: List of response curves for each channel
            budget: Total budget to allocate
            min_budgets: Minimum budget for each channel
            max_budgets: Maximum budget for each channel
            
        Returns:
            Dictionary containing:
            - 'allocation': Optimal budget allocation for each channel
            - 'response': Expected response for each channel
            - 'total_response': Total expected response
        """
        pass

class GeneticAlgorithmOptimizer:
    """Genetic algorithm implementation for budget optimization."""
    
    def __init__(self,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
    
    def _create_initial_population(self, n_channels: int, budget: float, 
                                 min_budgets: List[float], max_budgets: List[float]) -> np.ndarray:
        """Create initial population of budget allocations respecting constraints."""
        population = np.zeros((self.population_size, n_channels))
        
        for i in range(self.population_size):
            # Start with random proportions
            proportions = np.random.dirichlet(np.ones(n_channels))
            
            # Scale to total budget
            allocation = proportions * budget
            
            # Adjust to respect min/max constraints
            for j in range(n_channels):
                allocation[j] = max(min_budgets[j], min(max_budgets[j], allocation[j]))
            
            # Normalize remaining budget
            remaining = budget - allocation.sum()
            if remaining != 0:
                # Find channels that can take more budget
                can_increase = allocation < max_budgets
                if can_increase.any():
                    # Distribute remaining budget proportionally
                    proportions = np.random.dirichlet(np.ones(can_increase.sum()))
                    allocation[can_increase] += proportions * remaining
            
            population[i] = allocation
        
        return population
    
    def _evaluate_fitness(self, allocation: np.ndarray, curves: List[ResponseCurve],
                         min_budgets: List[float], max_budgets: List[float]) -> float:
        """Calculate total response for a given budget allocation."""
        # Check if allocation violates constraints
        if not self._is_valid_allocation(allocation, min_budgets, max_budgets):
            return float('-inf')
        
        total_response = 0
        for i, curve in enumerate(curves):
            total_response += curve.evaluate(allocation[i])
        return total_response
    
    def _is_valid_allocation(self, allocation: np.ndarray, 
                            min_budgets: List[float], max_budgets: List[float]) -> bool:
        """Check if an allocation satisfies all constraints."""
        for i in range(len(allocation)):
            if allocation[i] < min_budgets[i] or allocation[i] > max_budgets[i]:
                return False
        return True
    
    def _select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parents for next generation using tournament selection."""
        parents = np.zeros((self.population_size, population.shape[1]))
        for i in range(self.population_size):
            # Tournament selection
            candidates = np.random.choice(len(population), size=3, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            parents[i] = population[winner]
        return parents
    
    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover between parents to create offspring."""
        offspring = np.zeros_like(parents)
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
                # Single point crossover
                crossover_point = np.random.randint(1, parents.shape[1])
                offspring[i] = np.concatenate([parents[i][:crossover_point], 
                                            parents[i+1][crossover_point:]])
                offspring[i+1] = np.concatenate([parents[i+1][:crossover_point], 
                                              parents[i][crossover_point:]])
            else:
                offspring[i] = parents[i]
        return offspring
    
    def _mutate(self, population: np.ndarray, budget: float,
                min_budgets: List[float], max_budgets: List[float]) -> np.ndarray:
        """Apply mutations to the population while respecting constraints."""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                # Randomly adjust one channel's budget
                channel = np.random.randint(population.shape[1])
                adjustment = np.random.normal(0, budget * 0.1)  # 10% of budget as std dev
                
                # Apply mutation while respecting constraints
                new_value = population[i, channel] + adjustment
                new_value = max(min_budgets[channel], 
                              min(max_budgets[channel], new_value))
                
                # Calculate the difference to maintain total budget
                diff = new_value - population[i, channel]
                
                # Find other channels that can absorb the difference
                other_channels = np.arange(population.shape[1]) != channel
                can_adjust = other_channels & (
                    (diff < 0) |  # Can increase if we're decreasing the mutated channel
                    (population[i] < max_budgets)  # Can decrease if we're increasing the mutated channel
                )
                
                if can_adjust.any():
                    # Distribute the difference proportionally
                    proportions = np.random.dirichlet(np.ones(can_adjust.sum()))
                    population[i, can_adjust] -= proportions * diff
                    population[i, channel] = new_value
                    
                    # Ensure all constraints are still satisfied
                    population[i] = np.clip(population[i], 
                                          min_budgets, 
                                          max_budgets)
        
        return population
    
    def optimize(self, 
                curves: List[ResponseCurve],
                budget: float,
                min_budgets: List[float],
                max_budgets: List[float]) -> Dict[str, np.ndarray]:
        """Optimize using genetic algorithm."""
        n_channels = len(curves)
        
        # Validate total budget against constraints
        min_total = sum(min_budgets)
        max_total = sum(max_budgets)
        if budget < min_total:
            raise ValueError(f"Total budget {budget} is less than minimum required {min_total}")
        if budget > max_total:
            raise ValueError(f"Total budget {budget} exceeds maximum allowed {max_total}")
        
        # Initialize population
        population = self._create_initial_population(n_channels, budget, min_budgets, max_budgets)
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self._evaluate_fitness(ind, curves, min_budgets, max_budgets) 
                              for ind in population])
            
            # Update best solution
            generation_best_idx = np.argmax(fitness)
            if fitness[generation_best_idx] > best_fitness:
                best_fitness = fitness[generation_best_idx]
                best_solution = population[generation_best_idx].copy()
            
            # Select parents
            parents = self._select_parents(population, fitness)
            
            # Create offspring
            offspring = self._crossover(parents)
            
            # Apply mutations
            offspring = self._mutate(offspring, budget, min_budgets, max_budgets)
            
            # Elitism: preserve best solutions
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            population = np.vstack([population[elite_indices], offspring[:-self.elite_size]])
        
        # Calculate final responses
        channel_responses = np.array([curve.evaluate(best_solution[i]) 
                                    for i, curve in enumerate(curves)])
        
        return {
            'allocation': best_solution,
            'response': channel_responses,
            'total_response': best_fitness
        }

class GreedyIterativeOptimizer:
    """Greedy iterative allocation optimizer for local refinement."""
    
    def __init__(self, step_size: float = 0.01, max_iterations: int = 1000):
        self.step_size = step_size
        self.max_iterations = max_iterations
    
    def optimize(self,
                curves: List[ResponseCurve],
                budget: float,
                min_budgets: List[float],
                max_budgets: List[float],
                initial_allocation: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Optimize using greedy iterative allocation."""
        n_channels = len(curves)
        
        # Start with initial allocation or distribute budget evenly
        if initial_allocation is None:
            current_allocation = np.array([budget / n_channels] * n_channels)
        else:
            current_allocation = initial_allocation.copy()
        
        # Ensure initial allocation satisfies constraints
        current_allocation = np.clip(current_allocation, min_budgets, max_budgets)
        
        # Normalize to total budget
        total = current_allocation.sum()
        if total != budget:
            current_allocation = current_allocation * (budget / total)
        
        best_allocation = current_allocation.copy()
        best_response = self._evaluate_fitness(best_allocation, curves)
        
        for _ in range(self.max_iterations):
            improved = False
            
            # Try moving budget between each pair of channels
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    # Calculate step size for this iteration
                    step = self.step_size * budget
                    
                    # Try moving budget from i to j
                    new_allocation = current_allocation.copy()
                    new_allocation[i] -= step
                    new_allocation[j] += step
                    
                    # Ensure constraints are satisfied
                    if (new_allocation[i] >= min_budgets[i] and 
                        new_allocation[j] <= max_budgets[j]):
                        new_response = self._evaluate_fitness(new_allocation, curves)
                        if new_response > best_response:
                            best_allocation = new_allocation.copy()
                            best_response = new_response
                            improved = True
                    
                    # Try moving budget from j to i
                    new_allocation = current_allocation.copy()
                    new_allocation[j] -= step
                    new_allocation[i] += step
                    
                    # Ensure constraints are satisfied
                    if (new_allocation[j] >= min_budgets[j] and 
                        new_allocation[i] <= max_budgets[i]):
                        new_response = self._evaluate_fitness(new_allocation, curves)
                        if new_response > best_response:
                            best_allocation = new_allocation.copy()
                            best_response = new_response
                            improved = True
            
            if not improved:
                break
            
            current_allocation = best_allocation.copy()
        
        # Calculate final responses
        channel_responses = np.array([curve.evaluate(best_allocation[i]) 
                                    for i, curve in enumerate(curves)])
        
        return {
            'allocation': best_allocation,
            'response': channel_responses,
            'total_response': best_response
        }
    
    def _evaluate_fitness(self, allocation: np.ndarray, curves: List[ResponseCurve]) -> float:
        """Calculate total response for a given budget allocation."""
        total_response = 0
        for i, curve in enumerate(curves):
            total_response += curve.evaluate(allocation[i])
        return total_response

class BudgetOptimizer:
    """Optimizer for budget allocation across multiple channels."""
    
    def __init__(self, 
                 curves: List[ResponseCurve],
                 min_budgets: Optional[List[float]] = None,
                 max_budgets: Optional[List[float]] = None,
                 method: str = 'genetic',
                 **kwargs):
        """
        Initialize the budget optimizer.
        
        Args:
            curves: List of response curves for each channel
            min_budgets: Minimum budget for each channel (optional)
            max_budgets: Maximum budget for each channel (optional)
            method: Optimization method to use ('genetic' or 'greedy')
            **kwargs: Additional parameters for the specific optimization method
        """
        self.curves = curves
        self.n_channels = len(curves)
        
        # Set default min/max budgets if not provided
        self.min_budgets = min_budgets if min_budgets is not None else [0] * self.n_channels
        self.max_budgets = max_budgets if max_budgets is not None else [float('inf')] * self.n_channels
        
        # Validate constraints
        if len(self.min_budgets) != self.n_channels or len(self.max_budgets) != self.n_channels:
            raise ValueError("Number of min/max budgets must match number of channels")
        
        for i in range(self.n_channels):
            if self.min_budgets[i] > self.max_budgets[i]:
                raise ValueError(f"Min budget cannot be greater than max budget for channel {i}")
        
        # Initialize optimization method
        if method == 'genetic':
            self.optimizer = GeneticAlgorithmOptimizer(**kwargs)
        elif method == 'greedy':
            self.optimizer = GreedyIterativeOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def optimize(self, budget: float, **kwargs) -> Dict[str, np.ndarray]:
        """
        Optimize budget allocation across channels.
        
        Args:
            budget: Total budget to allocate
            **kwargs: Additional parameters for the optimization method
            
        Returns:
            Dictionary containing:
            - 'allocation': Optimal budget allocation for each channel
            - 'response': Expected response for each channel
            - 'total_response': Total expected response
        """
        return self.optimizer.optimize(
            curves=self.curves,
            budget=budget,
            min_budgets=self.min_budgets,
            max_budgets=self.max_budgets,
            **kwargs
        )
