import numpy as np
from abc import ABC, abstractmethod

# Define an abstract class for response curves
class ResponseCurve(ABC):
    """Base class for response curves."""
    
    # Abstract method must be used by any class that inherits this class
    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the response curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        pass

# A concrete class that inherits from the abstract class ResponseCurve
class GompertzCurve(ResponseCurve):
    """Gompertz response curve implementation."""
    
    def __init__(self, a: float, b: float, c: float):
        """
        Initialize Gompertz curve with parameters.
        
        Args:
            a: Upper asymptote
            b: Growth rate
            c: Inflection point
        """
        self.a = a
        self.b = b
        self.c = c
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Gompertz curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        return self.a * np.exp(-self.b * np.exp(-self.c * x))

class LogisticCurve(ResponseCurve):
    """Logistic (S-shaped) response curve implementation."""
    
    def __init__(self, L: float, k: float, x0: float):
        """
        Initialize Logistic curve with parameters.
        
        Args:
            L: Maximum value
            k: Growth rate
            x0: Midpoint
        """
        self.L = L
        self.k = k
        self.x0 = x0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Logistic curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        return self.L / (1 + np.exp(-self.k * (x - self.x0)))

class HillCurve(ResponseCurve):
    """Hill function response curve implementation."""
    
    def __init__(self, V: float, K: float, n: float):
        """
        Initialize Hill curve with parameters.
        
        Args:
            V: Maximum response
            K: Half-saturation point
            n: Hill coefficient (steepness)
        """
        self.V = V
        self.K = K
        self.n = n
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Hill curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        return (self.V * x**self.n) / (self.K**self.n + x**self.n)

class MichaelisMentenCurve(ResponseCurve):
    """Michaelis-Menten response curve implementation."""
    
    def __init__(self, V: float, K: float):
        """
        Initialize Michaelis-Menten curve with parameters.
        
        Args:
            V: Maximum response
            K: Half-saturation constant
        """
        self.V = V
        self.K = K
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Michaelis-Menten curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        return (self.V * x) / (self.K + x)

class WeibullCurve(ResponseCurve):
    """Weibull response curve implementation."""
    
    def __init__(self, a: float, b: float, c: float):
        """
        Initialize Weibull curve with parameters.
        
        Args:
            a: Maximum value
            b: Scale parameter
            c: Shape parameter
        """
        self.a = a
        self.b = b
        self.c = c
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Weibull curve at point(s) x.
        
        Args:
            x: Input values
            
        Returns:
            Response values
        """
        return self.a * (1 - np.exp(-(x/self.b)**self.c))
        
