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

### Option 1: Install from source
1. Clone the repository:
```bash
git clone https://github.com/yourusername/marketing_budget_optimizer.git
cd marketing_budget_optimizer
```

2. Install the package:
```bash
pip install -e .
```

### Option 2: Install directly from GitHub
```bash
pip install git+https://github.com/yourusername/marketing_budget_optimizer.git
```

## Usage

```python
from marketing_budget_optimizer import gompertz_curve

# Example usage
x = [1, 2, 3, 4, 5]
a, b, c = 100, 5, 0.5  # Example parameters
response = gompertz_curve(x, a, b, c)
```
