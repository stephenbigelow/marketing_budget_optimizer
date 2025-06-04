 import numpy as np
import pytest
from marketing_budget_optimizer.optimize import GeneticAlgorithmOptimizer

def test_optimizer_convergence(optimizer, test_curves):
    """Test that the optimizer converges to a good solution."""
    budget = 100000
    min_budgets = [1000] * len(test_curves)
    max_budgets = [50000] * len(test_curves)
    
    result = optimizer.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )
    
    # Test 1: Check that we get a valid allocation
    assert np.all(result['allocation'] >= min_budgets)
    assert np.all(result['allocation'] <= max_budgets)
    assert np.isclose(sum(result['allocation']), budget)
    
    # Test 2: Check convergence metrics
    metrics = result['convergence_metrics']
    assert metrics['final_improvement'] > 0  # Should improve from initial solution
    assert metrics['convergence_rate'] > 0  # Should show positive convergence rate
    assert metrics['population_diversity'] > 0  # Should maintain some diversity
    
    # Test 3: Check that fitness improves over generations
    best_fitness = result['best_fitness']
    assert best_fitness[-1] > best_fitness[0]  # Final solution should be better than initial
    
    # Test 4: Check that we get reasonable responses
    assert np.all(result['response'] >= 0)  # All responses should be non-negative
    assert result['total_response'] > 0  # Total response should be positive

def test_optimizer_parameter_sensitivity(optimizer, test_curves):
    """Test how different optimizer parameters affect convergence."""
    budget = 100000
    min_budgets = [1000] * len(test_curves)
    max_budgets = [50000] * len(test_curves)
    
    # Test with different population sizes
    small_pop = GeneticAlgorithmOptimizer(population_size=20, generations=30)
    large_pop = GeneticAlgorithmOptimizer(population_size=200, generations=30)
    
    small_result = small_pop.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )
    
    large_result = large_pop.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )
    
    # Larger population should generally find better solutions
    # but we allow for some variance due to stochastic nature
    assert large_result['total_response'] >= small_result['total_response'] * 0.95