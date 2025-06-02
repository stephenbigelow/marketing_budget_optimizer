import numpy as np
from abc import ABC, abstractmethod

class ResponseCurve(ABC):
    """Base class for response curves."""
    
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
