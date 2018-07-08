import numpy as np
import python.autodiff as ad


def test_add_by_const():
    x = ad.Variable("x")
    y = x + 1