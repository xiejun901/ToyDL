from python import back_prop as bp
import numpy as np
import math

def test_mul_two_var():
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

def test_matmul_two_var():
    x = bp.Variable("x")
    y = bp.Variable("y")
    z = bp.matmul_op(x, y)

    x_val = np.array([[1, 2, 3], [4, 5, 6]])
    y_val = np.array([[7, 8, 9, 10], [11, 12, 13,14], [15, 16, 17, 18]])
    z_val = np.matmul(x_val, y_val)

    excutor = bp.Executor()
    z_result, = excutor.forward([z], {x: x_val, y: y_val})
    assert np.array_equal(z_result, z_val)

    x_grad, y_grad = excutor.backward(z, [x, y], {x: x_val, y:y_val})
    z_grad = np.ones_like(z_result)

    expect_x_grad = np.matmul(z_grad, np.transpose(y_val))
    expect_y_grad = np.matmul(np.transpose(x_val), z_grad)

    assert np.array_equal(x_grad, expect_x_grad)
    assert np.array_equal(y_grad, expect_y_grad)


def test_exp_var():
    x = bp.Variable("x")
    y = bp.exp_op(x)

    x_val = np.array([1.0, 1.0])
    y_val = np.exp(x_val)

    executor = bp.Executor()

    y_result, = executor.forward([y], {x:x_val})

    assert np.array_equal(y_result, y_val)
    print(y_val)
    x_grad, = executor.backward(y, [x], {x:x_val})
    print(x_grad)
    np.testing.assert_almost_equal(x_grad, y_val)



