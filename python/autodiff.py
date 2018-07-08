import numpy as np


class Node(object):

    def __init__(self):
        self.name = ""
        self.op = None
        self.inputs = []
        self.const_attr = None

    def __add__(self, other):
        if isinstance(other, Node):
            return add_op(self, other)
        else:
            return add_byconst_op(self, other)

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul_op(self, other)
        else:
            return mul_byconst_op(self, other)

    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        return self.name

    __repr__ = __str__


class Op(object):

    def __call__(self):
        """
        通过运算符产生一个新的Node
        :return:
        """
        node = Node()
        node.op = self
        node.inputs = []
        return node

    def compute(self, node, input_vals):
        """
        前向计算
        :param node: 由此运算符计算出的节点
        :param input_vals:  输入值
        :return:
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """
        计算梯度
        :param node: 此运算符产生的节点
        :param output_grad: 输出的grad
        :return:
        """
        raise NotImplementedError

class PlaceholderOp(Op):

    def __call__(self):
        return Op.__call__(self)

    def compute(self, node, input_vals):
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        assert False, "never calculate gradient on placeholder"


class AddByConstOp(Op):

    def __call__(self, node, const_val):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.const_attr = const_val
        new_node.name = "({}+{})".format(node.name, const_val)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad]


class AddOp(Op):

    def __call__(self, node1, node2):
        new_node = Op.__call__(self)
        new_node.inputs = [node1, node2]
        new_node.name = "({}+{})".format(node1, node2)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]


class MulOp(Op):

    def __call__(self, node1, node2):
        new_node = Op.__call__(self)
        new_node.inputs = [node1, node2]
        new_node.name = "({}*{})".format(node1, node2)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def gradient(self, node, output_grad):
        return [node.inputs[0] * output_grad, output_grad * node.inputs[1]]


class MulByConstOp(Op):

    def __call__(self, node, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.name = f"({node}*{const_val})"
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def gradient(self, node, output_grad):
        return [output_grad * node.const_attr]


def Variable(name):
    node = placeholder_op()
    node.name = name
    return node


class Executor(object):
    """
    executor, 计算图的值和梯度
    """

    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        pass



add_op = AddOp()
add_byconst_op = AddByConstOp()
mul_op = MulOp()
mul_byconst_op = MulByConstOp()
placeholder_op = PlaceholderOp()
