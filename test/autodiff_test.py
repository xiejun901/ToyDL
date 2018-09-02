import numpy as np
import python.autodiff as ad


def test_add_by_const():
    x = ad.Variable("x")
    y = ad.Variable("y")
    z = ad.Variable("z")
    w = (x + y) * z + y
    executor= ad.Executor([w])
    result = executor.run({x:1, y:2, z:3})
    print(result)


def test_add_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
