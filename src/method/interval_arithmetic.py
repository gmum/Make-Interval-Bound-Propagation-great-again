import torch

class Interval:
    """
    A class representing an interval with lower and upper bounds, supporting interval arithmetic operations.
    Attributes:
        lower (torch.tensor): The lower bound of the interval.
        upper (torch.tensor): The upper bound of the interval.
    Methods:
        __init__(lower: torch.tensor, upper: torch.tensor):
            Initializes an Interval object with the given lower and upper bounds.
            Raises ValueError if the lower bound is greater than the upper bound.
        __add__(other: 'Interval') -> 'Interval':
            Adds two intervals or an interval and a scalar/tensor.
        __radd__(other: 'Interval') -> 'Interval':
            Adds two intervals or an interval and a scalar/tensor (right-hand addition).
        __sub__(other: 'Interval') -> 'Interval':
            Subtracts two intervals or an interval and a scalar/tensor.
        __rsub__(other: 'Interval') -> 'Interval':
            Subtracts two intervals or an interval and a scalar/tensor (right-hand subtraction).
        __mul__(other: 'Interval') -> 'Interval':
            Multiplies two intervals or an interval and a scalar/tensor.
        __rmul__(other: 'Interval') -> 'Interval':
            Multiplies two intervals or an interval and a scalar/tensor (right-hand multiplication).
        __truediv__(other: 'Interval') -> 'Interval':
            Divides two intervals or an interval and a scalar/tensor.
            Raises ZeroDivisionError if the divisor interval spans zero.
        __rtruediv__(other: 'Interval') -> 'Interval':
            Divides two intervals or an interval and a scalar/tensor (right-hand division).
            Raises ZeroDivisionError if the divisor interval spans zero.
        __matmul__(x) -> 'Interval':
            Performs matrix multiplication between an interval and a tensor with coefficients.
        __getitem__(idx):
            Retrieves a sub-interval at the specified index.
        __neg__() -> 'Interval':
            Negates the interval by flipping the signs of the bounds.
        diam() -> torch.tensor:
            Computes the diameter (width) of the interval.
        midpoint() -> torch.tensor:
            Computes the midpoint of the interval.
        right() -> 'Interval':
            Returns an interval where both bounds are equal to the upper bound.
        split() -> ('Interval', 'Interval'):
            Splits the interval into two parts: one centered at the midpoint and another representing the deviation.
        __repr__():
            Returns a string representation of the interval.
        __le__(value):
            Checks if the upper bound of the interval is less than or equal to a given value.
        __ge__(value):
            Checks if the lower bound of the interval is greater than or equal to a given value.
        exp() -> 'Interval':
            Computes the exponential of the interval bounds.
        sqr() -> 'Interval':
            Computes the square of the interval bounds.
        sum(dim=-1) -> 'Interval':
            Computes the sum of the interval bounds along the specified dimension.
    """

    def __init__(self, lower: torch.tensor, upper: torch.tensor):
        self.lower = lower
        self.upper = upper
        if torch.any(self.lower - 1e-8 > self.upper):
            raise ValueError("Lower bound must be less than or equal to the upper bound.")

    def __add__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        if isinstance(other, torch.Tensor):
            return Interval(self.lower + other, self.upper + other)
        if isinstance(other, float):
            return Interval(self.lower + other, self.upper + other)
        return NotImplemented
    
    def __radd__(self, other: 'Interval') -> 'Interval':
        return self + other

    def __sub__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        if isinstance(other, torch.Tensor):
            return Interval(self.lower - other, self.upper - other)
        if isinstance(other, float):
            return Interval(self.lower - other, self.upper - other)
        return NotImplemented
    
    def __rsub__(self, other: 'Interval') -> 'Interval':
        return -self + other

    def __mul__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, Interval):
            products = torch.stack([
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ])
            lower = torch.min(products, dim=0).values
            upper = torch.max(products, dim=0).values
            return Interval(lower, upper)
        if isinstance(other, torch.Tensor) or isinstance(other, float):
            products = torch.stack([self.lower * other, self.upper * other])
            lower = torch.min(products, dim=0).values
            upper = torch.max(products, dim=0).values
            return Interval(lower, upper)
        return NotImplemented
    
    def __rmul__(self, other: 'Interval') -> 'Interval':
        return self * other

    def __truediv__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, Interval):
            if torch.any(torch.logical_and(other.lower <= 0, 0 <= other.upper)):
                raise ZeroDivisionError("Division by an interval spanning zero is undefined.")
            reciprocals = torch.stack([
                1 / other.lower,
                1 / other.upper,
            ])
            return self * Interval(torch.min(reciprocals, dim=0).values, torch.max(reciprocals, dim=0).values)
        elif isinstance(other, torch.Tensor):
            return self * 1 / other
        return NotImplemented
    
    def __rtruediv__(self, other: 'Interval') -> 'Interval':
        if isinstance(other, Interval):
            if torch.any(torch.logical_and(self.lower <= 0, 0 <= self.upper)):
                raise ZeroDivisionError("Division by an interval spanning zero is undefined.")
            reciprocals = torch.stack([
                1 / self.lower,
                1 / self.upper,
            ])
            return Interval(torch.min(reciprocals, dim=0).values, torch.max(reciprocals, dim=0).values) * other
        elif isinstance(other, torch.Tensor):
            return Interval(1 / self.upper, 1 / self.lower) * other
        return NotImplemented
    
    def __matmul__(self, x) -> 'Interval':
        res_lower = torch.zeros(x.coeffs.shape)
        res_upper = torch.zeros(x.coeffs.shape)
        for i in range(self.lower.shape[0]):
            tmp = 0.0
            for j in range(x.coeffs.shape[0]):
                tmp += self[(i, j)] * x.coeffs[j]
            res_lower[i] = tmp.lower
            res_upper[i] = tmp.upper

        return Interval(res_lower, res_upper)
        
                
    
    def __getitem__(self, idx):
        return Interval(self.lower[idx], self.upper[idx])

    def __neg__(self) -> 'Interval':
        return Interval(-self.upper, -self.lower)

    def diam(self) -> torch.tensor:
        return self.upper - self.lower

    def midpoint(self) -> torch.tensor:
        return (self.lower + self.upper) / 2
    
    def right(self):
        return Interval(self.upper, self.upper)
    
    def split(self) -> 'Interval':
        mid_point = self.midpoint()
        r = self.upper - mid_point
        self.lower = mid_point
        self.upper = mid_point
        return mid_point, Interval(-r, r)
    
    def __repr__(self):
        lower = self.lower.ravel()
        upper = self.upper.ravel()    
        return f"Interval({torch.tensor(list(zip(lower, upper))).reshape((*self.lower.shape, 2))})"
    
    def __le__(self, value):
        return self.upper <= value
    
    def __ge__(self, value):
        return value <= self.lower
    
    def exp(self):
        return Interval(torch.exp(self.lower), torch.exp(self.upper))
    
    def sqr(self):
        return Interval(torch.square(self.lower), torch.square(self.upper))
    
    def sum(self, dim=-1):
        return Interval(self.lower.sum(dim=dim), self.upper.sum(dim=dim))
    

def interval_hull(i1: Interval, i2: Interval):
    lower = torch.minimum(i1.lower, i2.lower)
    upper = torch.maximum(i1.upper, i2.upper)
    return lower, upper

def intersection(i1: Interval, i2: Interval):
    lower = torch.where(i1.lower > i2.lower, i1.lower, i2.lower)
    upper = torch.where(i1.upper < i2.upper, i1.upper, i2.upper)
    return Interval(lower, upper)

#############################

def test_interval_creation():
    # Test valid interval creation
    i = Interval(lower=torch.tensor([1, 2]), upper=torch.tensor([3, 4]))
    assert (i.lower == torch.tensor([1, 2])).all()
    assert (i.upper == torch.tensor([3, 4])).all()

    # Test invalid interval creation (lower > upper)
    try:
        Interval(lower=torch.tensor([5, 6]), upper=torch.tensor([3, 4]))
        assert False, "Expected ValueError when lower > upper"
    except ValueError:
        pass


def test_interval_addition():
    # Interval + Interval
    i1 = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    i2 = Interval(torch.tensor([2, 1]), torch.tensor([4, 3]))
    result = i1 + i2
    assert (result.lower == torch.tensor([3, 3])).all()
    assert (result.upper == torch.tensor([7, 7])).all()

    # Interval + ndarray
    offset = torch.tensor([1, 1])
    result = i1 + offset
    assert (result.lower == torch.tensor([2, 3])).all()
    assert (result.upper == torch.tensor([4, 5])).all()


def test_interval_subtraction():
    i1 = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    i2 = Interval(torch.tensor([2, 1]), torch.tensor([4, 3]))

    # Interval - Interval
    result = i1 - i2
    assert (result.lower == torch.tensor([-3, -1])).all()
    assert (result.upper == torch.tensor([1, 3])).all()

    # Interval - ndarray
    offset = torch.tensor([1, 1])
    result = i1 - offset
    assert (result.lower == torch.tensor([0, 1])).all()
    assert (result.upper == torch.tensor([2, 3])).all()


def test_interval_multiplication():
    i1 = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    i2 = Interval(torch.tensor([2, 1]), torch.tensor([4, 3]))

    # Interval * Interval
    result = i1 * i2
    assert (result.lower == torch.tensor([2, 2])).all()
    assert (result.upper == torch.tensor([12, 12])).all()

    # Interval * ndarray
    multiplier = torch.tensor([2, 0.5])
    result = i1 * multiplier
    assert (result.lower == torch.tensor([2, 1])).all()
    assert (result.upper == torch.tensor([6, 2])).all()


def test_interval_division():
    i1 = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    i2 = Interval(torch.tensor([2, 1]), torch.tensor([4, 3]))

    # Interval / Interval
    result = i1 / i2
    assert (result.lower == torch.tensor([0.25, 2/3])).all()
    assert (result.upper == torch.tensor([1.5, 4])).all()

    # Division by interval spanning zero
    try:
        zero_spanning = Interval(torch.tensor([-1, -0.5]), torch.tensor([0.5, 1]))
        i1 / zero_spanning
        assert False, "Expected ZeroDivisionError for division by interval spanning zero"
    except ZeroDivisionError:
        pass


def test_interval_width_midpoint():
    i = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    # Test width
    assert (i.diam() == torch.tensor([2, 2])).all()
    # Test midpoint
    assert (i.midpoint() == torch.tensor([2, 3])).all()


def test_interval_negation():
    i = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    neg_i = -i
    assert (neg_i.lower == torch.tensor([-3, -4])).all()
    assert (neg_i.upper == torch.tensor([-1, -2])).all()


def test_interval_split():
    i = Interval(torch.tensor([1, 2]), torch.tensor([3, 4]))
    mid, q = i.split()

    # Test midpoint
    assert (mid == torch.tensor([2, 3])).all()
    # Test split ranges
    assert (q.lower == torch.tensor([-1, -1])).all()
    assert (q.upper == torch.tensor([1, 1])).all()
    # Confirm interval is updated
    assert (i.lower == mid).all()
    assert (i.upper == mid).all()


if __name__ == "__main__":
    test_interval_creation()
    test_interval_addition()
    test_interval_subtraction()
    test_interval_multiplication()
    test_interval_division()
    test_interval_width_midpoint()
    test_interval_negation()
    test_interval_split()

    print("All tests passed!")