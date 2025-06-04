import numpy as np
import pytest
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