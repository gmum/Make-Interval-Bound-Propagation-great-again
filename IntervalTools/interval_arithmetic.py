from __future__ import annotations

import torch
import numpy as np

from typing import Union

class Interval:
    """
        Basic interval implementation using Numpy package.
        Interval has a form [l,u] where l <= u.

        Attributes:
        -----------
            l: torch.Tensor or np.ndarray
                A lower bound of the interval.
            
            u: torch.Tensor or np.ndarray
                An upper bound of the interval.
        """
    
    def __init__(self, l: Union[torch.Tensor, np.ndarray], u: Union[torch.Tensor, np.ndarray]) -> None:

        assert l <= u, f"The upper bound should be greater or equal to the lower bound!"

        if isinstance(l, torch.Tensor):
          l = l.detach().cpu().numpy().item()
        if isinstance(u, torch.Tensor):
          u = u.detach().cpu().numpy().item()

        self.l = l
        self.u = u

    def intersection(self, other: Interval) -> Interval:
        """
        Calculates an intersection of two intervals.

        Parameters:
        -----------
            other: Interval
                Interval with which the intersection will be calculated.
        
        Returns:
        --------
            The calculated interval intersection.
        """
        if self.u < other.l or self.l > other.u:
            raise ValueError("Empty interval!")
        elif self <= other:
            return self
        elif self.u >= other.l and \
            self.u <= other.u:
            return Interval(other.l, self.u)
        elif other <= self:
            return other
        elif other.u >= self.l and \
            other.u <= self.u:
            return Interval(self.l, other.u)

    def __add__(self, other: Interval) -> Interval:
        """
        Addition of two intervals: [a,b] + [c,d] = [a+b, c+d].

        Parameters:
        -----------
            other: Interval
                Interval with which the addition will be calculated.
        
        Returns:
        --------
            The calculated interval addition.
        """
        return Interval(self.l + other.l, self.u + other.u)

    def __sub__(self, other: Interval) -> Interval:
        """
        Subtraction of two intervals: [a,b] - [c,d] = [a-d, b-c].

        Parameters:
        -----------
            other: Interval
                Interval with which the subtraction will be calculated.
        
        Returns:
        ---------
            The calculated interval subtraction.
        """
        return Interval(self.l - other.u, self.u - other.l)

    def __mul__(self, other: Interval) -> Interval:
        """
        Multiplication of two intervals: [a,b] * [c,d] = [min{ac,ad,bc,bd},max{ac,ad,bc,bd}].

        Parameters:
        -----------
            other: Interval
                Interval with which the multiplication will be calculated.

        Returns:
        --------
            The calculated interval multiplication.
        """
        vals = [self.l * other.l, self.l * other.u, self.u * other.l, self.u * other.u]
        lower, upper = min(vals), max(vals)
        return Interval(lower, upper)

    def __truediv__(self, other: Interval) -> Interval:
        """
        Division of two intervals. If the second interval contains 0, then division if undefined:
        [a,b] / [c,d] = [a,b] * [1/d, 1/c].

        Parameters:
        -----------
            other: Interval
                Interval with which the division will be calculated.
        
        Returns:
        --------
            The calculated interval division.
        """
        assert not (other.l == 0 or other.u == 0), f"Undefined division for {self} and {other}"
        assert not (other.l * other.u < 0), f"Undefined division for {self} and {other}"

        return self * Interval(1 / other.u, 1 / other.l)

    def __neg__(self) -> Interval:
        """
        Returns a negated interval.
        """
        return Interval(-self.u, -self.l)

    def __abs__(self) -> Interval:
        """
        Returns an absolute value interval.
        """
        l_min, u_max = np.minimum(np.abs(self.u), np.abs(self.l)), np.maximum(np.abs(self.u), np.abs(self.l))

        if self.l <= 0 and self.u >= 0:
          return Interval(0.0, u_max)
        else:
          return Interval(l_min, u_max)

    def __repr__(self) -> str:
        return f"[{self.l},{self.u}]"

    def __str__(self) -> str:
        return f"[{self.l:.4f},{self.u:.4f}]"

    def __le__(self, other: Interval) -> bool:
        """
        Checks if the 'other' Interval inlcudes 'self' Interval:
        [a, b] in [c, d] iff (a >= c and b <= d).

        Parameters:
        -----------
            other: Interval
                A superset interval.
            
        Returns:
        --------
            True or false value.
        """
        diff_l, diff_u = np.abs(self.l - other.l), np.abs(self.u - other.u)

        if diff_l > 1e-4:
            diff_l = 0.
        if diff_u > 1e-4:
            diff_u = 0.

        if self.l + diff_l >= other.l and self.u <= other.u + diff_u:
            return True
        return False
