# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import collections
from enum import Enum
import functools
import math
import multiprocessing as mp
import numpy as np
import os
import random
from typing import Dict, List, Tuple, TypeVar

class MCMCMethods(Enum):
    Metropolis = 'Metropolis method'
    Glauber = 'Glauber dynamics'
    SCA = 'Stochastic Cellular Automata'

class NodeType(Enum):
    Spin = frozenset({-1, +1})
    Binary = frozenset({0, 1})

NodeName = TypeVar('NodeName', int, str)

class IsingModel:
    """description of class"""
    def __init__(self, linear: Dict[NodeName, float], quadratic: Dict[Tuple[NodeName, NodeName], float]):
        self.__temperature: float = 0
        self.__pinningParameter: float = 0
        self.__nodeIndices: Dict[NodeName, int] = {  # 頂点の名前と__spinsの添字との対応。
            node: index
            for index, node in enumerate({n for pair in quadratic.keys() for n in pair}.union(linear.keys()))
        }
        self.__spins: List[int] = np.ones(len(self.__nodeIndices), dtype=np.int8)
        self.__externalMagneticFields: List[float] = np.array([
            linear[node] if node in linear else 0
            for node in self.__nodeIndices
        ], dtype=np.float)
        self.__couplingCoefficients: List[List[float]] = np.empty((len(self.__nodeIndices),)*2, dtype=np.float)
        for row, i in self.__nodeIndices.items():
            for column, j in self.__nodeIndices.items():
                if i == j:  # 対角成分は強制的に0にする。
                    self.__couplingCoefficients[i][j] = 0
                else:
                    if (row, column) in quadratic:
                        self.__couplingCoefficients[i][j] = quadratic[(row, column)]
                    elif (column, row) in quadratic:
                        self.__couplingCoefficients[i][j] = quadratic[(column, row)]
                    else:
                        self.__couplingCoefficients[i][j] = 0
        self.MarkovChain: MCMCMethods = MCMCMethods.Metropolis

    def CalcLocalMagneticField(self, nodeIndex: int, spins: List[int] = np.zeros((0, 0), dtype=np.int)) -> float:
        if spins.size == 0:
            spins = self.__spins
        return self.__externalMagneticFields[nodeIndex] + np.matmul(self.__couplingCoefficients, spins)[nodeIndex]

    @property
    def NodeIndices(self) -> Dict[NodeName, int]:
        return self.__nodeIndices

    @property
    def Spins(self) -> Dict[NodeName, int]:
        return {pair[0]: self.__spins[pair[1]] for pair in self.__nodeIndices.items()}

    @property
    def ExternalMagneticField(self) -> Dict[NodeName, float]:
        return {pair[0]: self.__externalMagneticFields[pair[1]] for pair in self.__nodeIndices.items()}

    @property
    def CouplingCoefficients(self) -> Dict[NodeName, Dict[NodeName, float]]:
        return {
            row[0]: {
                column[0]: self.__couplingCoefficients[row[1]][column[1]]
                for column in self.__nodeIndices.items()
            }
            for row in self.__nodeIndices.items()
        }

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

    def Print(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        maxColumns = math.ceil(math.sqrt(len(self.__spins)))
        for index in range(1, len(self.__spins) + 1):
            print(0 if self.__spins[index - 1] == -1 else 1, end='')
            if index % maxColumns == 0 or index == len(self.__spins):
                print()
            else:
                print(' ', end='')

    def Update(self):
        def metropolisMethod():
            updatedNode: int = random.randrange(len(self.__spins))
            energyDifference: float = 2.e0 * self.__spins[updatedNode] * self.CalcLocalMagneticField(updatedNode)
            if energyDifference < 0.e0:
                self.__spins[updatedNode] = -self.__spins[updatedNode]
            elif random.random() <= (np.exp(-energyDifference / self.__temperature) if self.__temperature != 0.e0 else 0.e0):
                self.__spins[updatedNode] = -self.__spins[updatedNode]

        def glauberDynamics():
            updatedNode: int = random.randrange(len(self.__spins))
            if random.random() <= 1.e0 / (1.e0 + np.exp(-2.e0 * self.CalcLocalMagneticField(updatedNode) / self.__temperature)):
                self.__spins[updatedNode] = +1
            else:
                self.__spins[updatedNode] = -1

        def sca():
            spins = self.__spins
            with mp.Pool(processes=8) as pool:
               self.__spins = np.array(pool.map(functools.partial(UpdateOneSpinForSCA, spins=spins, isingModel=self), self.__nodeIndices.keys()))
            #self.__spins = np.array(list(map(functools.partial(UpdateOneSpinForSCA, spins=spins, isingModel=self), self.__nodeIndices.keys())))

        if self.MarkovChain == MCMCMethods.Metropolis:
            metropolisMethod()
        elif self.MarkovChain == MCMCMethods.Glauber:
            glauberDynamics()
        elif self.MarkovChain == MCMCMethods.SCA:
            sca()
        else:
            raise ValueError('Illeagal choises')

    def GetEnergy(self):
        # Remove double-counting duplicates by multiplying the sum by 1/2.
        return 0.5e0 * np.matmul(-self.__spins, np.matmul(self.__couplingCoefficients, self.__spins) + self.__externalMagneticFields)

# 並列化の都合上、外部関数として定義する。
def UpdateOneSpinForSCA(nodeIndex: int, spins: List[int], isingModel: IsingModel) -> int:
    if random.random() <= 1.e0 / (1.e0 + np.exp((spins[nodeIndex] * isingModel.CalcLocalMagneticField(nodeIndex, spins) + isingModel.PinningParameter) / isingModel.Temperature)):
        return -spins[nodeIndex]
    else:
        return spins[nodeIndex]
