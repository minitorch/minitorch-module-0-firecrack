"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y


def id(x: float) -> float:
    """Return the input"""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers"""
    return x + y


def neg(x: float) -> float:
    """Return the negative of a number"""
    return -x


def lt(x: float, y: float) -> float:
    """Return 1.0 if x is less than y and 0.0 otherwise"""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Return 1.0 if x is equal to y and 0.0 otherwise"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Return 1.0 if x and y are less than 1e-2 apart, and 0.0 otherwise."""
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Compute the sigmoid function of x

    Args:
    ----
        x: A scalar or numpy array.

    Returns:
    -------
        1 / (1 + exp(-x))

    """
    return 1 / (1 + math.exp(-x))


def relu(x: float) -> float:
    """Compute the relu function of x

    Args:
    ----
        x: A scalar or numpy array.

    Returns:
    -------
        x if x is greater than or equal to 0, 0 otherwise

    """
    return x if x >= 0 else 0


def log(x: float) -> float:
    """Compute the natural logarithm of x

    Args:
    ----
        x: A scalar or numpy array.

    Returns:
    -------
        log(x)

    """
    return math.log(x)


def exp(x: float) -> float:
    """Compute the exp function of x

    Args:
    ----
        x: A scalar or numpy array.

    Returns:
    -------
        e^x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Compute the inverse function of x

    Args:
    ----
        x: A scalar or numpy array.

    Returns:
    -------
        1/x

    """
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Compute the gradient function for log
        ∂L/∂x = ∂L/∂ln(x) * (1/x) = y / x

    Args:
    ----
        x: A scalar or numpy array.
        y: A scalar, ∂L/∂ln(x)

    Returns:
    -------
        y / x

    """
    return y / x


def inv_back(x: float, y: float) -> float:
    """Compute the gradient function for inv
        ∂L/∂x = ∂L/∂(1/x) * (-x^-2) = -y / x^2

    Args:
    ----
        x: A scalar or numpy array.
        y: A scalar, ∂L/∂(1/x)

    Returns:
    -------
        -y / x^2

    """
    return -y / x**2


def relu_back(x: float, y: float) -> float:
    """Compute the gradient function for relu
        ∂L/∂x = ∂L/∂(ReLU(x)) * (0 if x < 0 else 1) = y * (0 if x < 0 else 1)

    Args:
    ----
        x: A scalar or numpy array.
        y: A scalar, ∂L/∂(ReLU(x))

    Returns:
    -------
        y if x >= 0 else 0

    """
    return y * (x >= 0)


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(a: Iterable[float], fn: Callable) -> Iterable[float]:
    """Map a function onto a list"""
    return [fn(x) for x in a]


def zipWith(
    fn: Callable[[float, float], float], a: Iterable[float], b: Iterable[float]
) -> Iterable[float]:
    """Combines two iterables element-wise using a specified function
    Args:
        fn: function to combine elements
        a: first iterable of floats, eg. list
        b: second iterable of floats, eg. list

    Returns
    -------
        A list of combined elements

    """
    return [fn(x, y) for x, y in zip(a, b)]


def reduce(
    a: Iterable[float], fn: Callable[[float, float], float], start: float = 0.0
) -> float:
    """Applies a function of two arguments cumulatively to the items of an
    iterable, so as to reduce the iterable to a single float value.

    Args:
    ----
        fn: function to apply
        a: iterable of numbers
        start: starting value

    Returns:
    -------
        A single number

    """
    res = start
    for i in a:
        res = fn(res, i)
    return res


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate all elements"""
    return map(ls, neg)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add two lists"""
    return zipWith(add, ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum up a list"""
    return reduce(ls, add, 0.0)


def prod(ls: Iterable[float]) -> float:
    """Multiply all elements"""
    return reduce(ls, mul, 1.0)
