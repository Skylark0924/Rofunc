# Robot Dynamics


## Overview

机器人的动力学方程是一个二阶微分方程

$$τ = M(θ) \ddot \theta+ h(θ, \dot \theta)$$

其中，
- $\theta$ 是关节变量的向量
- todo

**正向动力学**： 在给定状态 $\theta, \dot \theta$ 以及关节力和力矩的情况下，计算机器人的加速度 $\ddot \theta$ 

**逆向动力学**：在给定机器人状态以及需要的加速度的情况下，计算关节力和力矩。


机器人的动力学方程通常通过以下方式导出：
- 通过直接应用刚体的牛顿和欧拉动力学方程（通常称为牛顿-欧拉公式）
- 通过从机器人的动能和势能导出的拉格朗日动力学公式机器人

拉格朗日形式在概念上是优雅的，并且对于具有简单结构的机器人非常有效，例如具有三个或更少自由度的机器人。然而，对于自由度更高的机器人来说，计算很快就会变得很麻烦。对于一般的开链，Newton-Euler 公式为逆向和正向动力学提供了高效的递归算法，这些算法也可以组装成封闭形式的解析表达式，例如质量矩阵 M(θ) 和其他项动力学方程（8.1）。

## Lagrangian formulation
1. 选取一组独立的广义坐标$q$来描述系统；
2. 上述广义坐标可以用于定义广义力；
3. 拉格朗日方程 $\mathcal{L}(q, \dot q)$ 可以由系统的动能 $\mathcal{K}(q, \dot q)$ 和势能 $\mathcal{P}(q) $ 来定义，即为
$$ \mathcal{L}(q, \dot q)= \mathcal{K}(q, \dot q) -\mathcal{P}(q)$$
4. 运动方程现在可以用拉格朗日表示如下：
$$f=\dfrac{d}{dt}\dfrac{\partial\mathcal L}{\partial \dot q}-\dfrac{\partial \mathcal L}{\partial  q} $$



