import numpy as np
from marketing_budget_optimizer import GompertzCurve

# Test the GompertzCurve class
x = np.array([1, 2, 3, 4, 5])
a, b, c = 100, 5, 0.5

# Create curve instance
curve = GompertzCurve(a=a, b=b, c=c)
result = curve.evaluate(x)

print("Test successful!")
print(f"Input x: {x}")
print(f"Parameters: a={a}, b={b}, c={c}")
print(f"Result: {result}") 