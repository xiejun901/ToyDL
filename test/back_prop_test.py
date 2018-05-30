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


def test_relu_op():
    x = bp.Variable("x")
    y = bp.relu_op(x)

    x_val = np.array([[1, -2, 3], [-1, -1 ,3]])
    executor = bp.Executor()

    y_val, = executor.forward([y], {x: x_val})
    y_expect = np.array([[1, 0, 3], [0, 0, 3]])
    np.testing.assert_array_equal(y_val, y_expect)

    x_grad, = executor.backward(y, [x], {x:x_val})
    x_grad_expect = np.array([[1, 0, 1], [0, 0, 1]])
    np.testing.assert_array_equal(x_grad, x_grad_expect)

def test_sigmoid_op():
    x = bp.Variable("x")
    y = bp.sigmoid_op(x)

    x_val = np.array([1, 2])
    excecutor = bp.Executor()
    y_val, = excecutor.forward([y], feed_dict={x:x_val})
    y_expect = np.array([1/(1+math.exp(-1.0)), 1/(1+math.exp(-2.0))])
    np.testing.assert_almost_equal(y_val, y_expect)

    x_grad, = excecutor.backward(y, [x], {x:x_val})
    x_grad_expet = np.array([math.exp(-1.0)/(1+math.exp(-1.0))/(1+math.exp(-1.0)), math.exp(-2.0)/(1+math.exp(-2.0))/(1+math.exp(-2.0))])
    np.testing.assert_almost_equal(x_grad, x_grad_expet)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def test_sigmoid_cross_entropy_op():
    y = bp.Variable("y")
    label = bp.Variable("label")
    loss = bp.sigmoid_cross_entropy_op(y, label)
    y_val = np.array([1.0, 2.0, 3.0])
    label_val = np.array([4.0, 5.0, 6.0])
    loss_expect = -y_val*label_val+np.log(1.0 + np.exp(y_val))
    executor = bp.Executor()
    loss_val, = executor.forward([loss], feed_dict={y:y_val, label: label_val})
    np.testing.assert_almost_equal(loss_val, loss_expect)

    y_grad, label_grad = executor.backward(loss, [y, label], feed_dict={y:y_val, label: label_val})

    y_grad_expect = -label_val + sigmoid(y_val)
    label_grad_expect = -y_val
    np.testing.assert_almost_equal(y_grad, y_grad_expect)
    np.testing.assert_almost_equal(label_grad,label_grad_expect)


