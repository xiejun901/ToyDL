from python import back_prop as bp
import numpy as np

def test_mul_two_node():
    x = bp.Variable("x")
    y = bp.Variable("y")
    z = x * y
    excutor = bp.Executor()
    x_val = np.ones(3) * 3
    y_val = np.ones(3) * 5
    z_val, x_val_ = excutor.forward([z, x], {x: x_val, y: y_val})
    assert np.array_equal(x_val, x_val_)
    assert np.array_equal(z_val, x_val * y_val)

    x_grad, y_grad = excutor.backward(z, [x, y], {x: x_val, y: y_val})
    assert np.array_equal(x_grad, y_val)
    assert np.array_equal(y_grad, x_val)
