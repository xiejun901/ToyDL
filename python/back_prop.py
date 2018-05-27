
import numpy as np


class Node(object):

    def __init__(self):
        self.inputs = []
        self.outputs = []
        self.op = None
        self.name = ""

    def __add__(self, other):
        return add_op(self, other)

    def __mul__(self, other):
        return mul_op(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        return self.name

    __repr__ = __str__


class Op(object):

    def __call__(self):
        """
        调用运算符构造一个节点
        :return:
        """
        node = Node()
        node.op = self
        return node

    def compute(self, input_vals):
        """
        前向计算，根据输入的值计算出经过此运算符输出的值
        :param input_vals: 输入值 numpy.array
        :return:
        """
        raise NotImplementedError

    def bprop(self, input_vals, output_grad):
        """
        反向传播，根据输出的梯度和输入的值计算出输入的梯度
        :param input_vals:
        :param output_grad:
        :return:
        """
        raise NotImplementedError


def Variable(name):
    node = placeholder_op()
    node.name = name
    return node


class AddOp(Op):

    def __call__(self, node1, node2):
        node = Node()
        node.inputs = [node1, node2]
        node.op = self
        node.name = "{}+{}".format(node1.name, node2.name)
        node1.outputs.append(node)
        node2.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def bprop(self, input_vals, output_grad):
        return [output_grad, output_grad]


class MulOp(Op):

    def __call__(self, node1, node2):
        node = Node()
        node.inputs = [node1, node2]
        node.op = self
        node.name = "{}*{}".format(node1.name, node2.name)
        node1.outputs.append(node)
        node2.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def bprop(self, input_vals, output_grad):
        assert len(input_vals) == 2
        return [input_vals[1]*output_grad, input_vals[0]*output_grad]


class PlaceholderOp(Op):

    def __call__(self, *args, **kwargs):
        node = Op.__call__(self)
        return node

    def compute(self, input_vals):
        assert False, "placeholder values provided by feed_dict"

    def bprop(self, input_vals, output_grad):
        return Node


class MatMulOp(Op):

    def __call__(self, node1, node2):
        node = Node()
        node.inputs = [node1, node2]
        node.op = self
        node.name = "MatMulOp({},{})".format(node1.name, node2.name)
        node1.outputs.append(node)
        node2.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 2
        return np.matmul(input_vals[0], input_vals[1])

    def bprop(self, input_vals, output_grad):
        return [np.matmul(output_grad,np.transpose(input_vals[1])), np.matmul(np.transpose(input_vals[0]), output_grad)]


class ExpOp(Op):

    def __call__(self, node1):
        node = Node()
        node.inputs= [node1]
        node.op = self
        node.name = "exp({})".format(node1.name)
        node1.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 1
        return np.exp(input_vals[0])

    def bprop(self, input_vals, output_grad):
        return [np.exp(input_vals[0]) * output_grad]


class ReluOp(Op):

    def __call__(self, node1):
        node = Node()
        node.inputs = [node1]
        node.op = self
        node.name = "relu({})".format(node1.name)
        node1.ouputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 1
        return np.maximum(input_vals[0], 0)

    def bprop(self, input_vals, output_grad):
        return (np.sign(input_vals[0]) + 1)*0.5*output_grad


def sigmoid_func(x):
    temp = np.exp(x)
    return temp / (temp + 1)


class SigmoidOp(Op):

    def __call__(self, node1):
        node = Node()
        node.inputs = [node1]
        node.op = self
        node.name = "Sigmoid({})" / format(node1.name)
        node1.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 1
        return sigmoid_func(input_vals[0])

    def bprop(self, input_vals, output_grad):
        temp = sigmoid_func(input_vals[0])
        return [output_grad * (1 - temp) * temp]


class SigmoidCrossEntropyOp(Op):

    def __call__(self, logit, label):
        node = Node()
        node.inputs = [logit, label]
        node.op = self
        node.name = "SigmoidCrossEntropy({},{})".format(logit.name, label.name)
        logit.outputs.append(node)
        label.outputs.append(node)
        return node

    def compute(self, input_vals):
        assert len(input_vals) == 2
        sig = sigmoid_func(input_vals[0])
        return -1*(input_vals[1]*np.log(sig) + (1-input_vals[1])*np.log(1 - sig))

    def bprop(self, input_vals, output_grad):
        sig = sigmoid_func(input_vals[0])
        return ((1 - input_vals[1])*sig - input_vals[1]*(1-sig))*output_grad


add_op = AddOp()
mul_op = MulOp()
placeholder_op = PlaceholderOp()
matmul_op = MatMulOp()
exp_op = ExpOp()
relu_op = ReluOp()
sigmoid_op = SigmoidOp()
sigmoid_cross_entropy_op = SigmoidCrossEntropyOp()


class Executor(object):

    def __init__(self):
        self.node_to_val_map = {}
        self.node_to_grad_map = {}

    def forward(self, eval_node_list , feed_dict):
        self.node_to_val_map = dict(feed_dict)
        for node in eval_node_list:
            self.eval(node)
        return [self.node_to_val_map[n] for n in eval_node_list]

    def backward(self, output_node, grad_node_list, feed_dict):
        self.node_to_val_map = dict(feed_dict)
        self.node_to_grad_map = {}
        self.eval(output_node) # 计算输出节点及其依赖的节点的值
        self.node_to_grad_map[output_node] = np.ones_like(self.node_to_val_map[output_node])
        for node in grad_node_list:
            self.eval_grad(node)
        return [self.node_to_grad_map[n] for n in grad_node_list]

    def eval(self, node):
        if node in self.node_to_val_map:
            return
        for input_node in node.inputs:
            self.eval(input_node)
        input_vals = [self.node_to_val_map[input] for input in node.inputs]
        self.node_to_val_map[node] = node.op.compute(input_vals)

    def eval_grad(self, node):
        from functools import reduce
        if node in self.node_to_grad_map:
            return
        temp = []
        for output in node.outputs:
            if output not in self.node_to_val_map:
                """
                需要计算梯度的是被计算点的后续节点，和输出点的祖先节点，由于事先计算过输出节点的值，如果在值table里面不存在，则不是输出
                节点的祖先节点
                """
                continue
            self.eval_grad(output)
            input_vals = [self.node_to_val_map[n] for n in output.inputs]
            output_grad = self.node_to_grad_map[output]
            input_grads = output.op.bprop(input_vals, output_grad)
            for i in range(len(output.inputs)):
                if output.inputs[i] == node:
                    temp.append(input_grads[i])
        grad = reduce(lambda x1, x2: x1 + x2, temp)
        self.node_to_grad_map[node] = grad









