# ToyDL

学习深度学习过程中练习造的轮子

python/back_prop.py 节点，运算符的定义以及配套的图计算和反向传播算法，主要参考了书籍 [deeplearningbook](http://www.deeplearningbook.org/) 和 课程 [CSE 599W: Systems for ML](http://dlsys.cs.washington.edu/)

test/back_prop_test.py 反向传播模块的一些简单的test-case, 可以通过以下命令运行

```shell
nosetests test/back_prop_test.py
```

demo/linear_regression 简单的使用反向传播的线性回归的例子
demo/binary_classification 手动求导使用numpy的逻辑回归和使用反向传播的逻辑回归，mlp做二分类的例子


python/autodiff.py 自动求导的相关算法，梯度计算根据要求的节点生成对应的梯度图然后再进行计算

test/autodiff_test.py 自动求导模块的test-case, 部分测试直接使用了 [CSE 599W Assignment1](https://github.com/dlsys-course/assignment1) 
的测试用例

```shell
nosetests test/autodiff_test.py
```
```

