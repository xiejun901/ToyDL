
import numpy as np
import python.back_prop as bp

def get_data():
    w = np.array([1.])
    b = np.array([5.])
    x = np.random.rand(100)
    y = w * x + b
    return x, y

def main():
    x_val, y_val = get_data()
    x = bp.Variable("x")
    w = bp.Variable("w")
    b = bp.Variable("z")
    y = bp.Variable("z")

    y_pred = w * x + b
    loss = (y + -1 * y_pred)*(y + -1 * y_pred)

    w_val = np.random.rand(1)
    b_val = np.random.rand(1)

    executor = bp.Executor()

    i = 0
    n = 10000
    ln = 0.01
    while i < n:
        index = i % 100
        if(i % 20 == 0):
            print("step {}, w={}, b={}".format(i, w_val, b_val))
        w_grad, b_grad = executor.backward(loss, [w, b], feed_dict={x: x_val[index], y: y_val[index], w: w_val, b: b_val})
        w_val = w_val - ln*w_grad
        b_val = b_val - ln*b_grad
        i += 1


if __name__ == '__main__':
    main()