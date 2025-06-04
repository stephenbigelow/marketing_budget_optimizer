import numpy as np
from typing import List, Dict, Optional, Protocol
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
                 elite_size: int = 10,
                 tournament_size: int = 3): 
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        """
        population_size: Number of solutions in each generation
        generations: Number of generations to evolve
        mutation_rate: Probability of mutation for each gene
        elite_size: Number of best solutions to preserve in each generation
        tournament_size: Number of individuals to select for tournament selection
        """
    
    # This defines a random population of budget allocations respecting the desired constraints
    def _create_initial_population(self, n_channels: int, budget: float, 
                                 min_budgets: List[float], max_budgets: List[float]) -> np.ndarray:
        """Create initial population of budget allocations respecting constraints."""
        population = np.zeros((self.population_size, n_channels))
        
        for i in range(self.population_size):
            # Start with random proportions of budget for each channel
            # Dirichlet distribution ensures that the sum of the channel proportions is 1
            proportions = np.random.dirichlet(np.ones(n_channels))
            
            # Scale the total budget according to the proportions
            allocation = proportions * budget
            
            # Adjust to respect min/max constraints
            for j in range(n_channels):
                allocation[j] = max(min_budgets[j], min(max_budgets[j], allocation[j]))
            
            # Normalize remaining budget by randomly distributing it to channels that can take more budget
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
    
    # In genetic algorithms, fitness is the objective function that we want to maximize
    def _evaluate_fitness(self, allocation: np.ndarray, curves: List[ResponseCurve],
                         min_budgets: List[float], max_budgets: List[float]) -> float:
        """Calculate total response for a given budget allocation."""
        # Check if allocation violates constraints
        if not self._is_valid_allocation(allocation, min_budgets, max_budgets):
            # If the allocation violates constraints, return a very low fitness score
            # This will cause the allocation to be discarded in the next generation
            return float('-inf')
        
        # Otherwise, calculate the total response for the allocation
        total_response = 0
        for i, curve in enumerate(curves):
            total_response += curve.evaluate(allocation[i])
        return total_response
    
    # Helper method used in _evaluate_fitness to check if an allocation satisfies all constraints
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
        # Random selection of individuals (without replacement) for tournament, fittest wins
        # Repeat population_size times (note that previous tournament winners may be selected again)
        for i in range(self.population_size):
            # Tournament selection
            candidates = np.random.choice(len(population), size=self.tournament_size, replace=False)
            winner = candidates[np.argmax(fitness[candidates])]
            parents[i] = population[winner]
        return parents
    
    def _crossover(self, parents: np.ndarray) -> np.ndarray:
        """Perform crossover between parents to create offspring."""
        offspring = np.zeros_like(parents)
        # Performs single-point crossover to create 2 children from each pair of parents
        # Multi-point crossover is possible, but might be overly complex while disrupting good solutions
        # Potential todo: implement configurable number of crossover points
        for i in range(0, self.population_size, 2):
            if i + 1 < self.population_size:
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
                
                # Check if channels can be adjusted in the needed direction
                if diff > 0:
                    # If increasing mutated channel, other channels need to decrease
                    can_adjust = other_channels & (population[i] > min_budgets)
                else:
                    # If decreasing mutated channel, other channels need to increase
                    can_adjust = other_channels & (population[i] < max_budgets)
                
                if can_adjust.any():
                    # Distribute the difference proportionally
                    proportions = np.random.dirichlet(np.ones(can_adjust.sum()))
                    population[i, can_adjust] -= proportions * diff
                    population[i, channel] = new_value
                    
                    # Keep redistributing until budget is fully utilized
                    current_total = population[i].sum()
                    while abs(current_total - budget) >= 1e-10:  # Use small epsilon for float comparison
                        # Ensure all constraints are satisfied
                        population[i] = np.clip(population[i], 
                                              min_budgets, 
                                              max_budgets)
                        
                        # Update current total after clipping
                        current_total = population[i].sum()
                        
                        if current_total > budget:
                            # We have too much budget, need to reduce
                            excess = current_total - budget
                            # Find channels that can be reduced
                            can_decrease = population[i] > min_budgets
                            if not can_decrease.any():
                                # No channels can be reduced - this is a constraint violation
                                # We'll keep the current state and let the fitness function handle it
                                break
                            
                            # Distribute excess reduction proportionally
                            proportions = np.random.dirichlet(np.ones(can_decrease.sum()))
                            population[i, can_decrease] -= proportions * excess
                        else:
                            # We have too little budget, need to increase
                            deficit = budget - current_total
                            # Find channels that can take more budget
                            can_increase = population[i] < max_budgets
                            if not can_increase.any():
                                # No channels can be increased - this is a constraint violation
                                # We'll keep the current state and let the fitness function handle it
                                break
                            
                            # Distribute remaining budget proportionally
                            proportions = np.random.dirichlet(np.ones(can_increase.sum()))
                            population[i, can_increase] += proportions * deficit
                        
                        # Update current total after adjustments
                        current_total = population[i].sum()
        
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

class AdaptiveLocalOptimizer:
    """Local optimizer with adaptive step sizes and better exploration."""
    
    def __init__(self, 
                 initial_step_size: float = 0.01,
                 min_step_size: float = 0.001,
                 max_iterations: int = 1000,
                 improvement_threshold: float = 0.0001):
        self.initial_step_size = initial_step_size
        self.min_step_size = min_step_size
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold
    
    def optimize(self,
                curves: List[ResponseCurve],
                budget: float,
                min_budgets: List[float],
                max_budgets: List[float],
                initial_allocation: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Optimize using adaptive local search."""
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
        
        # Initialize step sizes for each channel pair
        step_sizes = np.full((n_channels, n_channels), self.initial_step_size)
        
        for iteration in range(self.max_iterations):
            improved = False
            current_step = self.initial_step_size * (1 - iteration / self.max_iterations)
            
            # Try moving budget between each pair of channels
            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    # Try multiple step sizes for this pair
                    for step_multiplier in [1.0, 0.5, 2.0]:
                        step = current_step * step_multiplier * budget
                        
                        # Try moving budget from i to j
                        new_allocation = current_allocation.copy()
                        new_allocation[i] -= step
                        new_allocation[j] += step
                        
                        # Ensure constraints are satisfied
                        if (new_allocation[i] >= min_budgets[i] and 
                            new_allocation[j] <= max_budgets[j]):
                            new_response = self._evaluate_fitness(new_allocation, curves)
                            if new_response > best_response * (1 + self.improvement_threshold):
                                best_allocation = new_allocation.copy()
                                best_response = new_response
                                improved = True
                                step_sizes[i, j] = step / budget  # Update successful step size
                        
                        # Try moving budget from j to i
                        new_allocation = current_allocation.copy()
                        new_allocation[j] -= step
                        new_allocation[i] += step
                        
                        # Ensure constraints are satisfied
                        if (new_allocation[j] >= min_budgets[j] and 
                            new_allocation[i] <= max_budgets[i]):
                            new_response = self._evaluate_fitness(new_allocation, curves)
                            if new_response > best_response * (1 + self.improvement_threshold):
                                best_allocation = new_allocation.copy()
                                best_response = new_response
                                improved = True
                                step_sizes[j, i] = step / budget  # Update successful step size
            
            if not improved:
                # If no improvement, try smaller steps
                current_step *= 0.5
                if current_step < self.min_step_size:
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
            method: Optimization method to use ('genetic', 'greedy', or 'adaptive')
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
        elif method == 'adaptive':
            self.optimizer = AdaptiveLocalOptimizer(**kwargs)
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
