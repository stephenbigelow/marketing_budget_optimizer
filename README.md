# Marketing Budget Optimizer

A Python package for optimizing marketing budgets using known response curves.

## Installation

### Prerequisites
Before installing this package, you must have:
1. Python 3.7 or higher
2. pip (Python's package installer)
3. setuptools (required for package installation)

Install setuptools first:
```bash
pip install setuptools
```

### Installation Options

#### Option 1: Install from source (Recommended for developers)
This method is best if you plan to modify the code or contribute to the project. The `-e` flag creates an "editable" install, meaning changes to the source code are immediately reflected without needing to reinstall.

1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketing_budget_optimizer.git
cd marketing_budget_optimizer
```

2. Install the package in editable mode:
```bash
pip install -e .
```

#### Option 2: Install directly from GitHub (Recommended for end users)
This method is simpler and suitable if you just want to use the package without modifying it:
```bash
pip install git+https://github.com/yourusername/marketing_budget_optimizer.git
```

## Usage

### Example: Optimizing Multi-Million Dollar Marketing Budget
This example demonstrates how to optimize a $10M marketing budget across multiple channels using realistic response curves:

```python
from marketing_budget_optimizer import GeneticAlgorithmOptimizer, HillCurve
import numpy as np

# Define marketing channels with realistic response curves
# Each curve represents a different channel's revenue response to marketing spend
channels = [
    # Channel 1: Quick initial growth, early inflection
    HillCurve(V=12_000_000, K=500_000, n=2.5),  # Lower K for earlier inflection
    
    # Channel 2: Very slow initial growth, very late inflection
    HillCurve(V=15_000_000, K=1_500_000, n=3.0),  # Higher K for later inflection
    
    # Channel 3: Moderate growth, mid-range inflection
    HillCurve(V=13_000_000, K=800_000, n=2.8),
    
    # Channel 4: Very quick initial growth, very early inflection
    HillCurve(V=11_000_000, K=300_000, n=2.2),  # Very low K for very early inflection
    
    # Channel 5: Extremely slow initial growth, extremely late inflection
    HillCurve(V=16_000_000, K=2_000_000, n=3.2),  # Very high K for very late inflection
    
    # Channel 6: Quick growth, mid-early inflection
    HillCurve(V=12_500_000, K=600_000, n=2.6),
    
    # Channel 7: Slow growth, mid-late inflection
    HillCurve(V=14_000_000, K=1_200_000, n=2.9),
    
    # Channel 8: Moderate growth, late inflection
    HillCurve(V=13_500_000, K=1_000_000, n=2.7)
]

# Define budget constraints for each channel
min_budgets = [
    300_000,  # Channel 1: Lower minimum for quick-return channel
    0,        # Channel 2: No minimum
    0,        # Channel 3: No minimum
    200_000,  # Channel 4: Very low minimum for very quick-return channel
    0,        # Channel 5: No minimum
    0,        # Channel 6: No minimum
    600_000,  # Channel 7: Higher minimum for quick-return channel
    0         # Channel 8: No minimum
]

max_budgets = [
    1_500_000,  # Channel 1: Lower maximum for quick-return channel
    float('inf'),  # Channel 2: No maximum
    float('inf'),  # Channel 3: No maximum
    1_200_000,  # Channel 4: Very low maximum for very quick-return channel
    float('inf'),  # Channel 5: No maximum
    float('inf'),  # Channel 6: No maximum
    2_200_000,  # Channel 7: Higher maximum for quick-return channel
    float('inf')   # Channel 8: No maximum
]

total_budget = 10_000_000  # $10M total budget

# Initialize the optimizer with custom parameters
optimizer = GeneticAlgorithmOptimizer(
    population_size=200,    # Larger population for better exploration
    generations=100,        # More generations for better convergence
    mutation_rate=0.1,      # 10% chance of mutation
    elite_size=20,         # Preserve top 20 solutions
    tournament_size=5      # Tournament selection size
)

# Run the optimization
result = optimizer.optimize(
    curves=channels,
    budget=total_budget,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    track_metrics=True  # Enable tracking of convergence metrics
)

# Print results
print("\nOptimal Budget Allocation:")
print("-" * 40)
for i, (channel, allocation) in enumerate(zip(channels, result['allocation'])):
    print(f"Channel {i+1}: ${allocation:,.2f}")
    print(f"Expected Revenue: ${result['response'][i]:,.2f}")
    print(f"ROI: {(result['response'][i] / allocation - 1) * 100:.1f}%")
    print()

print(f"Total Budget: ${result['allocation'].sum():,.2f}")
print(f"Total Expected Revenue: ${result['total_response']:,.2f}")
print(f"Overall ROI: {(result['total_response'] / total_budget - 1) * 100:.1f}%")

# Access convergence metrics
metrics = result['convergence_metrics']
print("\nOptimization Performance:")
print(f"Final Improvement: {metrics['final_improvement']:.2%}")
print(f"Convergence Rate: {metrics['convergence_rate']:.2f}")
print(f"Population Diversity: {metrics['population_diversity']:.2f}")
```

This example demonstrates:
1. Setting up response curves for different marketing channels using Hill curves
2. Defining budget constraints for each channel (some with no constraints)
3. Configuring the genetic algorithm optimizer
4. Running the optimization and analyzing results
5. Tracking convergence metrics

The optimizer will find the optimal budget allocation that maximizes total revenue while respecting:
- Minimum and maximum budget constraints for each channel (where specified)
- Total budget constraint
- Diminishing returns of marketing spend (modeled by Hill curves)
