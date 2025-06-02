import numpy as np
from typing import List, Dict, Tuple, Optional
from .curves import ResponseCurve

class BudgetOptimizer:
    """Genetic algorithm optimizer for budget allocation across multiple channels."""
    
    def __init__(self, 
                 curves: List[ResponseCurve],
                 min_budgets: Optional[List[float]] = None,
                 max_budgets: Optional[List[float]] = None,
                 population_size: int = 100,
                 generations: int = 50,
                 mutation_rate: float = 0.1,
                 elite_size: int = 10):
        """
        Initialize the budget optimizer.
        
        Args:
            curves: List of response curves for each channel
            min_budgets: Minimum budget for each channel (optional)
            max_budgets: Maximum budget for each channel (optional)
            population_size: Number of solutions in each generation
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation for each gene
            elite_size: Number of best solutions to preserve in each generation
        """
        self.curves = curves
        self.n_channels = len(curves)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Set default min/max budgets if not provided
        self.min_budgets = min_budgets if min_budgets is not None else [0] * self.n_channels
        self.max_budgets = max_budgets if max_budgets is not None else [float('inf')] * self.n_channels
        
        # Validate constraints
        if len(self.min_budgets) != self.n_channels or len(self.max_budgets) != self.n_channels:
            raise ValueError("Number of min/max budgets must match number of channels")
        
        for i in range(self.n_channels):
            if self.min_budgets[i] > self.max_budgets[i]:
                raise ValueError(f"Min budget cannot be greater than max budget for channel {i}")
    
    def _create_initial_population(self, budget: float) -> np.ndarray:
        """Create initial population of budget allocations respecting constraints."""
        population = np.zeros((self.population_size, self.n_channels))
        
        for i in range(self.population_size):
            # Start with random proportions
            proportions = np.random.dirichlet(np.ones(self.n_channels))
            
            # Scale to total budget
            allocation = proportions * budget
            
            # Adjust to respect min/max constraints
            for j in range(self.n_channels):
                allocation[j] = max(self.min_budgets[j], min(self.max_budgets[j], allocation[j]))
            
            # Normalize remaining budget
            remaining = budget - allocation.sum()
            if remaining != 0:
                # Find channels that can take more budget
                can_increase = allocation < self.max_budgets
                if can_increase.any():
                    # Distribute remaining budget proportionally
                    proportions = np.random.dirichlet(np.ones(can_increase.sum()))
                    allocation[can_increase] += proportions * remaining
            
            population[i] = allocation
        
        return population
    
    def _evaluate_fitness(self, allocation: np.ndarray) -> float:
        """Calculate total response for a given budget allocation."""
        # Check if allocation violates constraints
        if not self._is_valid_allocation(allocation):
            return float('-inf')
        
        total_response = 0
        for i, curve in enumerate(self.curves):
            total_response += curve.evaluate(allocation[i])
        return total_response
    
    def _is_valid_allocation(self, allocation: np.ndarray) -> bool:
        """Check if an allocation satisfies all constraints."""
        for i in range(self.n_channels):
            if allocation[i] < self.min_budgets[i] or allocation[i] > self.max_budgets[i]:
                return False
        return True
    
    def _select_parents(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Select parents for next generation using tournament selection."""
        parents = np.zeros((self.population_size, self.n_channels))
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
                crossover_point = np.random.randint(1, self.n_channels)
                offspring[i] = np.concatenate([parents[i][:crossover_point], 
                                            parents[i+1][crossover_point:]])
                offspring[i+1] = np.concatenate([parents[i+1][:crossover_point], 
                                              parents[i][crossover_point:]])
            else:
                offspring[i] = parents[i]
        return offspring
    
    def _mutate(self, population: np.ndarray, budget: float) -> np.ndarray:
        """Apply mutations to the population while respecting constraints."""
        for i in range(len(population)):
            if np.random.random() < self.mutation_rate:
                # Randomly adjust one channel's budget
                channel = np.random.randint(self.n_channels)
                adjustment = np.random.normal(0, budget * 0.1)  # 10% of budget as std dev
                
                # Apply mutation while respecting constraints
                new_value = population[i, channel] + adjustment
                new_value = max(self.min_budgets[channel], 
                              min(self.max_budgets[channel], new_value))
                
                # Calculate the difference to maintain total budget
                diff = new_value - population[i, channel]
                
                # Find other channels that can absorb the difference
                other_channels = np.arange(self.n_channels) != channel
                can_adjust = other_channels & (
                    (diff < 0) |  # Can increase if we're decreasing the mutated channel
                    (population[i] < self.max_budgets)  # Can decrease if we're increasing the mutated channel
                )
                
                if can_adjust.any():
                    # Distribute the difference proportionally
                    proportions = np.random.dirichlet(np.ones(can_adjust.sum()))
                    population[i, can_adjust] -= proportions * diff
                    population[i, channel] = new_value
                    
                    # Ensure all constraints are still satisfied
                    population[i] = np.clip(population[i], 
                                          self.min_budgets, 
                                          self.max_budgets)
        
        return population
    
    def optimize(self, budget: float) -> Dict[str, np.ndarray]:
        """
        Optimize budget allocation across channels.
        
        Args:
            budget: Total budget to allocate
            
        Returns:
            Dictionary containing:
            - 'allocation': Optimal budget allocation for each channel
            - 'response': Expected response for each channel
            - 'total_response': Total expected response
        """
        # Validate total budget against constraints
        min_total = sum(self.min_budgets)
        max_total = sum(self.max_budgets)
        if budget < min_total:
            raise ValueError(f"Total budget {budget} is less than minimum required {min_total}")
        if budget > max_total:
            raise ValueError(f"Total budget {budget} exceeds maximum allowed {max_total}")
        
        # Initialize population
        population = self._create_initial_population(budget)
        best_solution = None
        best_fitness = float('-inf')
        
        for generation in range(self.generations):
            # Evaluate fitness
            fitness = np.array([self._evaluate_fitness(ind) for ind in population])
            
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
            offspring = self._mutate(offspring, budget)
            
            # Elitism: preserve best solutions
            elite_indices = np.argsort(fitness)[-self.elite_size:]
            population = np.vstack([population[elite_indices], offspring[:-self.elite_size]])
        
        # Calculate final responses
        channel_responses = np.array([curve.evaluate(best_solution[i]) 
                                    for i, curve in enumerate(self.curves)])
        
        return {
            'allocation': best_solution,
            'response': channel_responses,
            'total_response': best_fitness
        }
