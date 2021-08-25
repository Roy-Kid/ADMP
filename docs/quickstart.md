---
title: 快速开始
date: 2021-08-24
sidebar: 'auto'
---

项目根目录下有多个子文件夹:

* python: 使用python实现的力场原型, 快速开发, 验证理论正确性;
* tests: 单元测试, 测试函数输入输出的;

## python原型

`python`文件夹下有:

* ADMPForce.py: 入口文件;
* neighborList.py: 临近表构建;
* pme.py: 力场计算;
* utils.py: 通用工具

代码的构建是围绕`ADMPForce.py`中`ADMPGenerator`和`ADMPBaseForce`类进行的. `ADMPGenerator`是个工厂类, 负责所有与系统信息有关的**静态**参数的读取处理与构建. `ADMPForce`类是实际计算中操纵的类, 提供一系列的计算接口. `Generator`在处理完参数之后, 实例化一个`Force`, 将在整个计算过程中不再改变的参数传给实例. 调用实例的接口可以计算一系列的能量和力, 得到粒子的信息. 这些信息会交由i-pi/OpenMM等分子动力学积分器进行运算, 再将更新后的位置传回`Force`. 每一步迭代后, `Force`都需要更新相关的参数, 以便能量的计算. 

![workflow](\workflow.png)

在未接入积分器的版本中, pdb/xml文件的读取由`read_mpid_inputs()`进行, 经过处理之后传给`ADMPForce`. `ADMPForce`接收到参数之后, 首先要调用`update()`方法计算动态的参数(与位置有关的临近表, kappa及K矢, frame转换等), 再调用`calc_real_space_energy()`等计算能量. 

```python

force = generator.create_force()
force.update()
print(force.kappa) # 0.32
print(force.calc_real_space_energy()) # 878.869

```

## 测试构建

在实现过程中为了满足预期的输入输出, 后期修改时保证对其它功能不产生影响, 我们需要自动化的测试工具以覆盖大多数的代码. 对于python的测试, 我们使用了[pytest](https://docs.pytest.org/en/6.2.x/index.html)测试框架.

在`tests`文件夹中的`conftest.py`文件中构建模型, 也就是我们需要的测试用例. 例如, 我要从`waterdimer_aligned.pdb`出发, 构建测试用的二水模型.

```python
@pytest.fixture  #指出接下来的函数是一个测试用例
def water_dimer():
    rc = 8 # in Angstrom
    ethresh = 1e-4
    pdb = 'tests/samples/waterdimer_aligned.pdb'
    xml = 'tests/samples/mpidwater.xml'
    gen = ADMPGenerator(pdb, xml, rc, ethresh...)    
    
    # yield以上为测试用例的构建,
    # 例如参数准备, 文件创建等.
    yield gen  # yield返回创建好的ADMPGenerator
    # 用例使用完成后返回并执行一下代码,
    # 进行用例的清理工作, 如临时文件的删除.
    # 可以以相同参数调用MPID等标准文件以检验结果正确性
```
`conftest.py`文件在执行pytest时首先被载入, 不需要也不能进行`import`. 在`test_pme.py`中, 我们要实际测试代码的结果是否符合预期. 

```python
# 测试需要以test_开头
# water即conftest.py中的用例名
def test_water_dimer(water_dimer):
    generator = water_dimer
    force = generator.create_force()
    force.update()
    force.kappa = 0.328532611
    assert force.kappa == 0.328532611
    real_e = force.calc_real_space_energy()
    assert real_e == 878.8693561091234
    # 使用assert来判断返回值与预期是否相同
```