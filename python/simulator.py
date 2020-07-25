# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

from enum import Enum, auto
import numpy as np
import numpy.linalg as LA
from typing import Dict, List, Optional, Tuple, Union  # TypeVarとUnionのどちらが良いのか？　ラベルの型を混ぜられるようにするためにUnionを使う？

class Algorithms(Enum):
    Metropolis = 'Metropolis method'
    Glauber = 'Glauber dynamics'
    SCA = 'Stochastic Cellular Automata'
    MA = 'Momentum Annealing'

class ConfigurationsType(Enum):
    AllDown = auto()
    AllUp = auto()
    Uniform = auto()

class NodeType(Enum):
    Spin = frozenset({-1, +1})
    Binary = frozenset({0, 1})

#NodeName = TypeVar('NodeName', int, str)
NodeName = Union[int, str]

class IsingModel:
    """An Ising model simulator"""
    def __init__(self, linear: Dict[NodeName, float], quadratic: Dict[Tuple[NodeName, NodeName], float]):
        self.__rng: np.random.Generator = np.random.default_rng()
        self.__temperature: float = 0  # 温度パラメータ。
        self.__pinningParameter: float = 0  # SCAのpinning parameter.
        self.__nodeLabels: List[NodeName] = [  # __spinsの添字と頂点の名前との対応。
            node
            for node in {n for pair in quadratic.keys() for n in pair}.union(linear.keys())
        ]
        self.__spins: List[int] = np.ones(len(self.__nodeLabels), dtype=np.int8)
        self.__externalMagneticField: List[float] = np.array([  # 外部磁場の強さ。
            linear[self.__nodeLabels[index]]
            if self.__nodeLabels[index] in linear else 0
            for index in range(len(self.__nodeLabels))
        ], dtype=np.float)
        self.__couplingCoefficients: List[List[float]] = np.zeros((len(self.__nodeLabels),)*2, dtype=np.float)  # スピン同士の結合定数（対角成分が0の対称行列）。
        for row in range(len(self.__nodeLabels)):
            for column in range(row, len(self.__nodeLabels)):
                if (self.__nodeLabels[row], self.__nodeLabels[column]) in quadratic:
                    self.__couplingCoefficients[row][column] = self.__couplingCoefficients[column][row] = quadratic[(self.__nodeLabels[row], self.__nodeLabels[column])]
                else:
                    self.__couplingCoefficients[row][column] = self.__couplingCoefficients[column][row] = 0.e0
        self.Algorithm: Algorithms = Algorithms.Metropolis  # 使用する更新アルゴリズム。
        self.__previousSpins: List[int] = self.__spins

    def __calcLocalMagneticFieldAt(self, nodeIndex: int) -> float:
        return self.__externalMagneticField[nodeIndex] + np.matmul(self.__couplingCoefficients, self.__spins)[nodeIndex]

    def __calcLocalMagneticField(self, spins: List[int]=np.zeros((0, 0), dtype=np.int)) -> List[float]:
        if spins.size == 0:
            spins = self.__spins
        return self.__externalMagneticField + np.matmul(self.__couplingCoefficients, spins)

    @property
    def Spins(self) -> Dict[NodeName, int]:
        return {node: self.__spins[i] for i, node in enumerate(self.__nodeLabels)}

    @property
    def Temperature(self) -> float:
        return self.__temperature

    @Temperature.setter
    def Temperature(self, temperature: float):
        self.__temperature = max(temperature, 0.e0)  # 強制的に非負実数にする。

    @property
    def PinningParameter(self) -> float:
        return self.__pinningParameter

    @PinningParameter.setter
    def PinningParameter(self, pinningParameter: float):
        self.__pinningParameter = max(pinningParameter, 0.e0)  # 強制的に非負実数にする。

    @property
    def Energy(self) -> float:
        def hamiltonian() -> float:
            # Remove double-counting duplicates by multiplying the sum by 1/2.
            return -0.5e0 * np.matmul(self.__spins, np.matmul(self.__couplingCoefficients, self.__spins))\
                - np.inner(self.__spins, self.__externalMagneticField)

        def hamiltonianOnBipartiteGraph() -> float:
            return -0.5e0 * np.matmul(self.__spins, np.matmul(self.__couplingCoefficients, self.__previousSpins))\
                - 0.5e0 * np.inner(self.__spins + self.__previousSpins, self.__externalMagneticField)\
                + 0.5e0 * self.__pinningParameter * (self.__spins.size - np.inner(self.__spins, self.__previousSpins))

        if self.Algorithm == Algorithms.Metropolis:
            return hamiltonian()
        elif self.Algorithm == Algorithms.Glauber:
            return hamiltonian()
        elif self.Algorithm == Algorithms.SCA:
            return hamiltonianOnBipartiteGraph()
        elif self.Algorithm == Algorithms.MA:
            return hamiltonianOnBipartiteGraph()
        else:
            raise ValueError('Illeagal choises')

    def CalcLargestEigenvalue(self) -> float:
        return np.amax(LA.eigvalsh(-self.__couplingCoefficients))

    def GiveSpins(self, confType: ConfigurationsType):
        if confType == ConfigurationsType.AllDown:
            self.__spins = np.full(self.__spins.size, -1)
        elif confType == ConfigurationsType.AllUp:
            self.__spins = np.ones(self.__spins.size)
        elif confType == ConfigurationsType.Uniform:
            self.__spins = self.__rng.choice([-1, 1], self.__spins.size)
        else:
            raise ValueError('Illeagal choises')
        self.__previousSpins = self.__spins

    def SetSeed(self, seed: Optional[int]=None):  # numpy.randomのドキュメントを見るに、int以外も渡せるようにするべき？
        self.__rng = np.random.default_rng(seed)

    def Update(self):
        def metropolisMethod():
            updatedNode: int = self.__rng.integers(len(self.__spins))
            energyDifference: float = 2.e0 * self.__spins[updatedNode] * self.__calcLocalMagneticFieldAt(updatedNode)
            if energyDifference < 0.e0:
                self.__spins[updatedNode] = -self.__spins[updatedNode]
            elif self.__rng.random() <= (np.exp(-energyDifference / self.__temperature) if self.__temperature != 0.e0 else 0.e0):
                self.__spins[updatedNode] = -self.__spins[updatedNode]

        def glauberDynamics():
            updatedNode: int = self.__rng.integers(len(self.__spins))
            if self.__rng.random() <= 1.e0 / (1.e0 + np.exp(-2.e0 * self.__calcLocalMagneticFieldAt(updatedNode) / self.__temperature)):
                self.__spins[updatedNode] = +1
            else:
                self.__spins[updatedNode] = -1

        # あるいはProbabilistic Cellular Automata.
        def stochasticCellularAutomata():
            size = len(self.__nodeLabels)
            self.__previousSpins = self.__spins
            self.__spins = np.sign(  # 実質起こらないが、符号関数に渡しているため、スピンが0になる場合がある。
                self.__externalMagneticField
                + np.matmul(self.__couplingCoefficients + np.identity(size) * self.__pinningParameter, self.__spins)
                - self.__temperature * self.__rng.logistic(size=size)
            )

        # 温度を下げなければ ``annealing'' ではないが、論文では区別していないので、ここでもこの名称を用いる。
        def momentumAnnealing():
            size = len(self.__nodeLabels)
            temp = np.sign(  # 実質起こらないが、符号関数に渡しているため、スピンが0になる場合がある。
                self.__externalMagneticField
                + np.matmul(self.__couplingCoefficients + np.identity(size) * self.__pinningParameter, self.__spins)
                - self.__temperature * (self.__rng.exponential(size=size) * self.__previousSpins)
            )
            self.__previousSpins = self.__spins
            self.__spins = temp

        if self.Algorithm == Algorithms.Metropolis:
            metropolisMethod()
        elif self.Algorithm == Algorithms.Glauber:
            glauberDynamics()
        elif self.Algorithm == Algorithms.SCA:
            stochasticCellularAutomata()
        elif self.Algorithm == Algorithms.MA:
            momentumAnnealing()
        else:
            raise ValueError('Illeagal choises')

    def Write(self):
        print('Current spin configuration:')
        print(self.__spins)
        print('External magnetic field:')
        print(self.__externalMagneticField)
        print('Coupling coefficinets:')
        print(self.__couplingCoefficients)
        print('Algorithm:', self.Algorithm.value)
        print('Temperature:', self.Temperature)
        print('Pinning parameter:', self.PinningParameter)
