# Import necessary modules and functions
import pytest
from imstat import add


# Define test functions
def test_addition():
    assert add(1, 2) == 3


def test_addition_negative_numbers():
    assert add(-1, -2) == -3


def test_addition_floats():
    assert add(1.5, 2.5) == 4.0
