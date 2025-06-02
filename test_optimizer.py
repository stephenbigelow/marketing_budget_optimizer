import numpy as np
from marketing_budget_optimizer import GompertzCurve, BudgetOptimizer

# Create example response curves for different channels
curves = [
    GompertzCurve(a=100, b=5, c=0.5),    # Channel 1: High asymptote, moderate growth
    GompertzCurve(a=80, b=4, c=0.6),     # Channel 2: Lower asymptote, slower growth
    GompertzCurve(a=120, b=6, c=0.4),    # Channel 3: Highest asymptote, fastest growth
]

# Define channel-specific budget constraints
min_budgets = [50, 100, 200]    # Minimum required budget for each channel
max_budgets = [200, 300, 800]   # Maximum allowed budget for each channel

# Initialize optimizer with custom parameters and constraints
optimizer = BudgetOptimizer(
    curves=curves,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    population_size=500,  # Larger population for better exploration
    generations=50,       # More generations for better convergence
    mutation_rate=0.1,    # 10% chance of mutation
    elite_size=10        # Preserve top 10 solutions
)

# Optimize budget allocation
total_budget = 1000
result = optimizer.optimize(total_budget)

# Print detailed results
print("Channel Budget Constraints:")
print("-" * 40)
for i in range(len(curves)):
    print(f"Channel {i+1}:")
    print(f"  Min Budget: ${min_budgets[i]:.2f}")
    print(f"  Max Budget: ${max_budgets[i]:.2f}")
print()

print("Optimal Budget Allocation:")
print("-" * 40)
for i, (allocation, response) in enumerate(zip(result['allocation'], result['response'])):
    print(f"Channel {i+1}:")
    print(f"  Budget: ${allocation:.2f}")
    print(f"  Expected Response: {response:.2f}")
    print(f"  Response per Dollar: {response/allocation:.4f}")
    print(f"  Within Constraints: {min_budgets[i] <= allocation <= max_budgets[i]}")
    print()

print(f"Total Budget Used: ${result['allocation'].sum():.2f}")
print(f"Total Expected Response: {result['total_response']:.2f}")
print(f"Average Response per Dollar: {result['total_response']/total_budget:.4f}")

# Verify budget constraint
print("\nBudget Constraint Check:")
print(f"Total Budget Used: ${result['allocation'].sum():.2f}")
print(f"Budget Constraint: ${total_budget:.2f}")
print(f"Constraint Satisfied: {abs(result['allocation'].sum() - total_budget) < 1e-10}")

# Verify channel-specific constraints
print("\nChannel-Specific Constraint Check:")
all_constraints_satisfied = True
for i, allocation in enumerate(result['allocation']):
    if not (min_budgets[i] <= allocation <= max_budgets[i]):
        all_constraints_satisfied = False
        print(f"Channel {i+1} constraint violated: ${allocation:.2f} not in [${min_budgets[i]:.2f}, ${max_budgets[i]:.2f}]")
print(f"All channel constraints satisfied: {all_constraints_satisfied}") 