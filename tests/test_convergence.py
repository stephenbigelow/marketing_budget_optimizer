import numpy as np
import pytest
import matplotlib.pyplot as plt
from marketing_budget_optimizer.optimize import GeneticAlgorithmOptimizer
from marketing_budget_optimizer.curves import ResponseCurve

class TestResponseCurve(ResponseCurve):
    """Simple test response curve for testing."""
    def __init__(self, a: float, b: float):
        self.a = a
        self.b = b
    
    def evaluate(self, x: float) -> float:
        """Simple diminishing returns curve: a * (1 - exp(-b * x))"""
        return self.a * (1 - np.exp(-self.b * x))

@pytest.fixture
def test_curves():
    """Create test response curves with different characteristics."""
    return [
        TestResponseCurve(a=1000, b=0.001),  # High ceiling, slow saturation
        TestResponseCurve(a=500, b=0.002),   # Medium ceiling, medium saturation
        TestResponseCurve(a=200, b=0.005)    # Low ceiling, fast saturation
    ]

@pytest.fixture
def optimizer():
    """Create a genetic algorithm optimizer with test parameters."""
    return GeneticAlgorithmOptimizer(
        population_size=50,
        generations=30,
        mutation_rate=0.1,
        elite_size=5
    )

@pytest.fixture
def optimization_result(optimizer, test_curves):
    """Run optimization and get results with metrics."""
    budget = 100000
    min_budgets = [1000] * len(test_curves)
    max_budgets = [50000] * len(test_curves)
    return optimizer.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )

def test_convergence_metrics(optimization_result):
    """Test that convergence metrics are calculated correctly."""
    convergence_metrics = optimization_result['convergence_metrics']
    
    # Check that all required metrics are present
    assert 'final_improvement' in convergence_metrics
    assert 'convergence_rate' in convergence_metrics
    assert 'population_diversity' in convergence_metrics
    assert 'best_fitness' in convergence_metrics
    
    # Check that metrics have reasonable values
    assert convergence_metrics['best_fitness'] > 0
    assert convergence_metrics['population_diversity'] >= 0
    assert isinstance(convergence_metrics['convergence_rate'], float)

def test_fitness_monotonicity(optimization_result):
    """Test that best fitness never decreases."""
    best_fitness = optimization_result['best_fitness']
    
    # Check that best fitness is monotonically increasing
    for i in range(1, len(best_fitness)):
        assert best_fitness[i] >= best_fitness[i-1]

def test_allocation_constraints(optimization_result):
    """Test that allocations always satisfy constraints."""
    best_allocations = optimization_result['best_allocation']
    budget = optimization_result['allocation'].sum()  # Total budget from final allocation
    
    # Check each generation's best allocation
    for allocation in best_allocations:
        # Check budget sum
        assert np.isclose(np.sum(allocation), budget, rtol=1e-5)
        
        # Check min/max constraints (using the same constraints as in the fixture)
        min_budgets = [1000] * len(allocation)
        max_budgets = [50000] * len(allocation)
        for i, (alloc, min_b, max_b) in enumerate(zip(allocation, min_budgets, max_budgets)):
            assert min_b <= alloc <= max_b

def test_population_diversity(optimization_result):
    """Test that population diversity decreases over time."""
    std_fitness = optimization_result['std_fitness']
    
    # Check that standard deviation generally decreases
    # (allowing for some noise in early generations)
    early_std = np.mean(std_fitness[:5])
    late_std = np.mean(std_fitness[-5:])
    assert late_std <= early_std

def test_convergence_visualization(optimization_result):
    """Test that convergence plots can be generated."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot fitness metrics
    generations = range(len(optimization_result['best_fitness']))
    ax1.plot(generations, optimization_result['best_fitness'], 'b-', label='Best Fitness')
    ax1.fill_between(generations,
                    optimization_result['best_fitness'] - optimization_result['std_fitness'],
                    optimization_result['best_fitness'] + optimization_result['std_fitness'],
                    alpha=0.2, color='b')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Convergence')
    ax1.legend()
    ax1.grid(True)
    
    # Plot allocation convergence
    best_allocations = optimization_result['best_allocation']
    for i in range(best_allocations.shape[1]):
        ax2.plot(generations, best_allocations[:, i], label=f'Channel {i+1}')
    
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Budget Allocation')
    ax2.set_title('Budget Allocation Convergence')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Check that figure has correct number of subplots
    assert len(fig.axes) == 2
    
    # Check that plots have correct titles
    assert fig.axes[0].get_title() == 'Fitness Convergence'
    assert fig.axes[1].get_title() == 'Budget Allocation Convergence'
    
    plt.close(fig)

def test_multiple_runs_consistency(optimizer, test_curves):
    """Test that multiple runs produce similar results."""
    budget = 100000
    min_budgets = [1000] * len(test_curves)
    max_budgets = [50000] * len(test_curves)
    
    result1 = optimizer.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )
    
    result2 = optimizer.optimize(
        curves=test_curves,
        budget=budget,
        min_budgets=min_budgets,
        max_budgets=max_budgets,
        track_metrics=True
    )
    
    # Check that final best fitness values are within 10% of each other
    final_fitness1 = result1['best_fitness'][-1]
    final_fitness2 = result2['best_fitness'][-1]
    assert abs(final_fitness1 - final_fitness2) / final_fitness1 < 0.1

def test_early_convergence(optimization_result):
    """Test that significant improvement happens in early generations."""
    best_fitness = optimization_result['best_fitness']
    
    # Check that 50% of total improvement happens in first 30% of generations
    total_improvement = best_fitness[-1] - best_fitness[0]
    early_improvement = best_fitness[int(len(best_fitness) * 0.3)] - best_fitness[0]
    assert early_improvement >= 0.5 * total_improvement 