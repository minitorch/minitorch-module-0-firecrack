import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generate N random points

    Arags:
      N: Number of points to generate

    Returns
    -------
      A list of N point

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """生成一个简单的二分类数据集图形对象。
    该函数根据给定的点数N, 创建一个基于x_1坐标值的二分类数据集。
    当x_1坐标小于0.5时, 分类标签为1; 否则为0。

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """生成分割线为对角线的二维数据集

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """生成一个具有非线性标签边界的二维分类数据集。

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """生成一个具有XOR边界的二维分类数据集。

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """生成一个具有圆形边界的二维分类数据集。

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """生成一个具有螺旋的二维分类数据集。

    参数:
        N (int): 要生成的数据点数量

    返回:
        Graph: 包含N个数据点、坐标集合X和对应标签y的图形对象

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
