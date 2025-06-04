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
from marketing_budget_optimizer import GeneticAlgorithmOptimizer, GompertzCurve
import numpy as np

# Define marketing channels with realistic response curves
# Each curve represents a different channel's revenue response to marketing spend
channels = [
    # Channel 1: Paid Search (High ceiling, moderate growth)
    GompertzCurve(a=50_000_000, b=2.5, c=0.00002),  # $50M max revenue
    
    # Channel 2: Social Media (Medium ceiling, faster growth)
    GompertzCurve(a=30_000_000, b=3.0, c=0.00003),  # $30M max revenue
    
    # Channel 3: Display Ads (Lower ceiling, rapid saturation)
    GompertzCurve(a=20_000_000, b=3.5, c=0.00004),  # $20M max revenue
    
    # Channel 4: Video Marketing (High ceiling, slow growth)
    GompertzCurve(a=40_000_000, b=2.0, c=0.00001),  # $40M max revenue
]

# Define budget constraints for each channel
min_budgets = [500_000, 300_000, 200_000, 400_000]  # Minimum required spend
max_budgets = [4_000_000, 3_000_000, 2_000_000, 3_000_000]  # Maximum allowed spend
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
1. Setting up realistic response curves for different marketing channels
2. Defining budget constraints for each channel
3. Configuring the genetic algorithm optimizer
4. Running the optimization and analyzing results
5. Tracking convergence metrics

The optimizer will find the optimal budget allocation that maximizes total revenue while respecting:
- Minimum and maximum budget constraints for each channel
- Total budget constraint
- Diminishing returns of marketing spend (modeled by Gompertz curves)
