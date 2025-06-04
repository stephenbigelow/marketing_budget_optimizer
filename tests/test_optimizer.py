import numpy as np
from marketing_budget_optimizer import GompertzCurve, BudgetOptimizer

def print_results(result, title):
    """Helper function to print optimization results."""
    print(f"\n{title}")
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

# Create example response curves for different channels
curves = [
    GompertzCurve(a=1000000, b=2, c=0.00005),    
    GompertzCurve(a=8000000, b=4, c=0.00002),    
    GompertzCurve(a=1200000, b=3, c=0.00007),    
    GompertzCurve(a=2500000, b=2.5, c=0.00004),  # Medium ceiling, moderate growth
    GompertzCurve(a=5000000, b=3.5, c=0.00003),  # High ceiling, steeper growth
    GompertzCurve(a=900000, b=2.8, c=0.00006),   # Lower ceiling, faster saturation
    GompertzCurve(a=3000000, b=3.2, c=0.00004),  # Medium-high ceiling, balanced growth
    GompertzCurve(a=1500000, b=2.2, c=0.00008)   # Medium-low ceiling, rapid saturation
]

# Define channel-specific budget constraints
min_budgets = [10000, 80000, 12000, 25000, 50000, 9000, 30000, 15000]    # Minimum required budget for each channel
max_budgets = [200000, 1000000, 240000, 500000, 800000, 180000, 600000, 300000]   # Maximum allowed budget for each channel
total_budget = 2500000

print("Channel Budget Constraints:")
print("-" * 40)
for i in range(len(curves)):
    print(f"Channel {i+1}:")
    print(f"  Min Budget: ${min_budgets[i]:.2f}")
    print(f"  Max Budget: ${max_budgets[i]:.2f}")
print()

# 1. Genetic Algorithm Optimization
print("\n1. Running Genetic Algorithm Optimization...")
genetic_optimizer = BudgetOptimizer(
    curves=curves,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    method='genetic',
    population_size=500,  # Larger population for better exploration
    generations=50,       # More generations for better convergence
    mutation_rate=0.1,    # 10% chance of mutation
    elite_size=10        # Preserve top 10 solutions
)

genetic_result = genetic_optimizer.optimize(total_budget)
print_results(genetic_result, "Genetic Algorithm Results")

# 2. Greedy Iterative Optimization
print("\n2. Running Greedy Iterative Optimization...")
greedy_optimizer = BudgetOptimizer(
    curves=curves,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    method='greedy',
    step_size=0.01,      # 1% of total budget as step size
    max_iterations=1000  # Maximum number of iterations
)

greedy_result = greedy_optimizer.optimize(total_budget)
print_results(greedy_result, "Greedy Optimization Results")

# 3. Combined Approach: Genetic + Greedy
print("\n3. Running Combined Optimization (Genetic + Greedy)...")
# First use genetic algorithm for global search
combined_optimizer = BudgetOptimizer(
    curves=curves,
    min_budgets=min_budgets,
    max_budgets=max_budgets,
    method='greedy',
    step_size=0.005,     # Smaller step size for finer refinement
    max_iterations=2000  # More iterations for better refinement
)

# Use genetic algorithm result as initial point for greedy optimization
combined_result = combined_optimizer.optimize(
    total_budget,
    initial_allocation=genetic_result['allocation']
)
print_results(combined_result, "Combined Optimization Results")

# Compare results
print("\nComparison of Results:")
print("-" * 40)
print(f"Genetic Algorithm Total Response: {genetic_result['total_response']:.2f}")
print(f"Greedy Optimization Total Response: {greedy_result['total_response']:.2f}")
print(f"Combined Approach Total Response: {combined_result['total_response']:.2f}")

# Verify budget constraints
print("\nBudget Constraint Check:")
print(f"Total Budget Used: ${combined_result['allocation'].sum():.2f}")
print(f"Budget Constraint: ${total_budget:.2f}")
print(f"Constraint Satisfied: {abs(combined_result['allocation'].sum() - total_budget) < 1e-10}")

# Verify channel-specific constraints
print("\nChannel-Specific Constraint Check:")
all_constraints_satisfied = True
for i, allocation in enumerate(combined_result['allocation']):
    if not (min_budgets[i] <= allocation <= max_budgets[i]):
        all_constraints_satisfied = False
        print(f"Channel {i+1} constraint violated: ${allocation:.2f} not in [${min_budgets[i]:.2f}, ${max_budgets[i]:.2f}]")
print(f"All channel constraints satisfied: {all_constraints_satisfied}") 