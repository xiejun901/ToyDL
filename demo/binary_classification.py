
import numpy as np
import python.back_prop as bp


def load_libsvm(path):
    pair_feature = []
    label = []
    feature = []
    with open(path) as f:
        for line in f:
            splits = line.strip().split(" ")
            label.append([float(splits[0])])
            temp = {}
            for kv in splits[1:]:
                k, v = kv.strip().split(":")
                temp[int(k)] = int(v)
            pair_feature.append(temp)
        max_key = 0
        for fe in pair_feature:
            for k in fe.keys():
                if k > max_key:
                    max_key = k
        for fe in pair_feature:
            temp = [0.0] * max_key
            for k, v in fe.items():
                temp[k-1] = float(v)
            # temp.append(1.0)
            feature.append(temp)
    return np.array(label), np.array(feature)


def lr():
    label, feature = load_libsvm("data/agaricus.txt")
    print("total example: {}".format(len(label)))

    x = bp.Variable("x")
    w = bp.Variable("w1")
    b = bp.Variable("b1")
    y = bp.Variable("y")

    y_pred = bp.matmul_op(x, w) + b

    prob = bp.sigmoid_op(y_pred)

    single_loss = bp.sigmoid_cross_entropy_op(logit=y_pred, label=y)

    w_val = np.random.rand(126, 1)
    b_val = np.random.rand(1, 1)

    ln = 0.0001

    excutor = bp.Executor()

    for i in range(1000000):
        index = i % len(feature)
        if i % 1000 == 0:
            loss_val, = excutor.forward([single_loss], feed_dict={
                x: feature,
                w: w_val,
                b: b_val,
                y: label
            })
            prob_val, = excutor.forward([prob], feed_dict={
                x: feature,
                w: w_val,
                b: b_val,
                y: label
            })
            print("step {}, loss={}, acc={}".format(i, np.mean(loss_val), cal_acc(label, prob_val)))

        w1_grad, b1_grad = excutor.backward(single_loss, [w, b], feed_dict={
            x: feature[index:index + 1],
            w: w_val,
            b: b_val,
            y: label[index:index + 1]
        })
        w_val = w_val - ln * w1_grad
        b_val = b_val - ln * b1_grad

def mlp():
    label, feature = load_libsvm("data/agaricus.txt")

    x = bp.Variable("x")
    w1 = bp.Variable("w1")
    b1 = bp.Variable("b1")
    w2 = bp.Variable("w2")
    b2 = bp.Variable("b2")
    y = bp.Variable("y")

    h1 = bp.relu_op(bp.matmul_op(x, w1) + b1)
    y_pred = bp.matmul_op(h1, w2) + b2
    prob = bp.sigmoid_op(y_pred)
    single_loss = bp.sigmoid_cross_entropy_op(logit=y_pred, label = y)

    w1_val = np.random.rand(126, 32)
    b1_val = np.random.rand(1, 32)
    w2_val = np.random.rand(32, 1)
    b2_val = np.random.rand(1, 1)

    ln = 0.001

    excutor = bp.Executor()

    for i in range(10000000):
        index = i % len(feature)
        if i % 1000 == 0:
            loss_val, prob_val = excutor.forward([single_loss, prob], feed_dict={
                x: feature,
                w1: w1_val,
                b1: b1_val,
                w2: w2_val,
                b2: b2_val,
                y: label
            })
            print("step {}, loss={}, acc={}, ln={}".format(i, np.mean(loss_val), cal_acc(label, prob_val), ln))
        if i % 500000 == 0:
            ln = ln / 10
        w1_grad, b1_grad, w2_grad, b2_grad = excutor.backward(single_loss, [w1, b1, w2, b2], feed_dict={
            x: feature[index:index+1],
            w1: w1_val,
            b1: b1_val,
            w2: w2_val,
            b2: b2_val,
            y: label[index:index+1]
        })
        w1_val = w1_val - ln * w1_grad
        b1_val = b1_val - ln * b1_grad
        w2_val = w2_val - ln * w2_grad
        b2_val = b2_val - ln * b2_grad


def lr_np():
    label, feature = load_libsvm("data/agaricus.txt")
    # 如果使用lr_np，对特征最后加上一列1.0,在运算的时候就不用计算b了
    W = np.random.rand(127, 1)
    for i in range(1000):
        index = i%len(feature)
        x = feature
        sigmod = bp.sigmoid_func(np.dot(x, W))
        x_t = x.transpose()
        y = label
        dw = np.dot(-1 * x_t, y) + np.dot(x_t, sigmod)
        W = W - 0.0001*dw
        acc = cal_acc(y, sigmod)
        if i % 200 == 0:
            print("acc={}".format(acc))






def cal_acc(label, pred):
    corrected = np.sum(np.equal(np.round(pred).astype(int), label.astype(int)))
    total = len(label)
    return corrected / float(total)

if __name__ == '__main__':
    mlp()


            

