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


class OnesLikeOp(Op):
    """
    产生与输入节点shape相同的全1元素的张量
    """

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = f"OnesLike(${node.name})"
        return new_node

    def compute(self, node, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeros_like_op(output_grad[0])]


class ZerosLikeOp(Op):
    """
    产生与输入节点shape相同的全0元素的张量
    """

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = f"ZerosLike(${node.name})"
        return new_node

    def compute(self, node, input_vals):
        assert (isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return zeros_like_op(node)


class PlaceholderOp(Op):

    def __call__(self):
        return Op.__call__(self)

    def compute(self, node, input_vals):
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        return None


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
        new_node.inputs = [node]
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
        node_to_value = dict(feed_dict)
        node_list = topology_sort(self.eval_node_list)
        for node in node_list:
            if node not in node_to_value:
                input_vals = list(map(lambda x: node_to_value[x], node.inputs))
                value = node.op.compute(node, input_vals)
                node_to_value[node] = value
        return list(map(lambda x: node_to_value[x], self.eval_node_list))


def gradients(output_node, node_list):
    """
    根据输出节点生成对node_list中节点的梯度
    :param output_node:
    :param node_list:
    :type output_node: Node
    :type node_list: list[Node]
    :return:
    """

    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [ones_like_op(output_node)]
    reverse_topo_order = reversed(topology_sort([output_node]))
    node_to_output_grad = {}
    for node in reverse_topo_order:
        """
        对于每一个节点，由于按拓扑排序逆许进行计算，那么其所有后续路径已经计算完毕，在这个地方可以计算输出相对于节点的梯度
        同时，更新前依赖节点本条路径的梯度
        """
        node_to_output_grad[node] = sum_node_list(node_to_output_grads_list[node])
        input_grad = node.op.gradient(node, node_to_output_grad[node])
        for i in range(len(node.inputs)):
            n = node.inputs[i]
            g = input_grad[i]
            if n not in node_to_output_grads_list:
                node_to_output_grads_list[n] = [g]
            else:
                node_to_output_grads_list[n].append(g)
    return [node_to_output_grad[n] for n in node_list]


def sum_node_list(node_list):
    """
    对节点进行求和
    :param node_list:
    :return:
    """
    from functools import reduce
    return reduce(lambda n1, n2: n1 + n2, node_list)


def topology_sort(node_list):
    """
    找到node_list 中所有node的依赖node
    :param node_list:
    :type node_list list[Node]
    :return:
    """
    result = []
    visited = set([])

    def dfs(node):

        for n in node.inputs:
            if n not in visited:
                dfs(n)
            else:
                pass
        result.append(node)
        visited.add(node)

    for node in node_list:
        dfs(node)
    return result


add_op = AddOp()
add_byconst_op = AddByConstOp()
mul_op = MulOp()
mul_byconst_op = MulByConstOp()
placeholder_op = PlaceholderOp()
ones_like_op = OnesLikeOp()
zeros_like_op = ZerosLikeOp()
