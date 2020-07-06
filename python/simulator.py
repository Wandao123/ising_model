# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

from enum import Enum
import multiprocess as mp  # 標準ライブラリと異なることに注意。
import numpy as np
import numpy.linalg as LA
import random
from typing import Dict, List, Tuple, TypeVar

class Algorithms(Enum):
    Metropolis = 'Metropolis method'
    Glauber = 'Glauber dynamics'
    SCA = 'Stochastic Cellular Automata'

class NodeType(Enum):
    Spin = frozenset({-1, +1})
    Binary = frozenset({0, 1})

NodeName = TypeVar('NodeName', int, str)

class IsingModel:
    """An Ising model simulator"""
    def __init__(self, linear: Dict[NodeName, float], quadratic: Dict[Tuple[NodeName, NodeName], float]):
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
        self.Parallelizing: bool = False  # SCAでPythonによる並列化をするか否かのフラグ。

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
        # Remove double-counting duplicates by multiplying the sum by 1/2.
        return np.matmul(-self.__spins, 0.5e0 * np.matmul(self.__couplingCoefficients, self.__spins) + self.__externalMagneticField)

    def CalcLargestEigenvalue(self) -> float:
        return np.amax(LA.eigvalsh(-self.__couplingCoefficients))

    def Update(self):
        def metropolisMethod():
            updatedNode: int = random.randrange(len(self.__spins))
            energyDifference: float = 2.e0 * self.__spins[updatedNode] * self.__calcLocalMagneticFieldAt(updatedNode)
            if energyDifference < 0.e0:
                self.__spins[updatedNode] = -self.__spins[updatedNode]
            elif random.random() <= (np.exp(-energyDifference / self.__temperature) if self.__temperature != 0.e0 else 0.e0):
                self.__spins[updatedNode] = -self.__spins[updatedNode]

        def glauberDynamics():
            updatedNode: int = random.randrange(len(self.__spins))
            if random.random() <= 1.e0 / (1.e0 + np.exp(-2.e0 * self.__calcLocalMagneticFieldAt(updatedNode) / self.__temperature)):
                self.__spins[updatedNode] = +1
            else:
                self.__spins[updatedNode] = -1

        def stochasticCellularAutomata():
            NumProcesses = 8
            spins = self.__spins
            localMagneticField = self.__calcLocalMagneticField(spins)

            def updateOneSpin(updatedSpin: int, localMagneticFieldAt: float) -> int:
                if random.random() <= 1.e0 / (1.e0 + np.exp((updatedSpin * localMagneticFieldAt + self.__pinningParameter) / self.__temperature)):
                    return -updatedSpin
                else:
                    return updatedSpin

            def updateSpinsFor(queue: mp.Queue, nodeIndicesArray: List[int]) -> List[int]:
                result = np.empty(nodeIndicesArray.size, dtype=np.int)
                print(nodeIndicesArray)
                for i in range(result.size):
                    if random.random() <= 1.e0 / (1.e0 + np.exp((spins[nodeIndicesArray[i]] * localMagneticField[nodeIndicesArray[i]] + self.__pinningParameter) / self.__temperature)):
                        result[i] = -spins[nodeIndicesArray[i]]
                    else:
                        result[i] = spins[nodeIndicesArray[i]]
                queue.put(result)
                return result

            # Warning: Python側で並列処理を実装しているものの、実際はNumpyに直接渡した方が速い。
            if self.Parallelizing:
                # multiprocess.Poolを使う方法。
                with mp.Pool(processes=NumProcesses) as pool:
                   self.__spins = np.array(pool.map(lambda nodeIndex: updateOneSpin(spins[nodeIndex], localMagneticField[nodeIndex]), range(len(self.__nodeLabels))))

                # multiprocess.Processesを使う方法。
                #queue = mp.Queue()
                #splitNodeIndicesArray = np.array_split(np.arange(len(self.__nodeLabels)), NumProcesses)  # 各プロセスへパメータの組を適当に振り分ける。
                #processesList = []
                #for i in range(NumProcesses):
                #    process = mp.Process(target=updateSpinsFor, args=(queue, splitNodeIndicesArray[i]))
                #    process.start()
                #    processesList.append(process)
                #self.__spins = np.concatenate([queue.get() for i in range(NumProcesses)])
            else:
                self.__spins = np.array([
                    updateOneSpin(spins[nodeIndex], localMagneticField[nodeIndex])
                    for nodeIndex in range(len(self.__nodeLabels))
                ], dtype=np.int)

        if self.Algorithm == Algorithms.Metropolis:
            metropolisMethod()
        elif self.Algorithm == Algorithms.Glauber:
            glauberDynamics()
        elif self.Algorithm == Algorithms.SCA:
            stochasticCellularAutomata()
        else:
            raise ValueError('Illeagal choises')

    def Write(self):
        print('Current spin configuration:')
        print(self.__spins)
        print()
        print('External magnetic field:')
        print(self.__externalMagneticField)
        print('Coupling coefficinets:')
        print(self.__couplingCoefficients)
        print('Algorithm:', self.Algorithm.value)
