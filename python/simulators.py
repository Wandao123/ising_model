# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import collections
from enum import Enum
import functools
import math
import multiprocessing as mp
from numba import jit, jitclass, types, typed, typeof, deferred_type
#import numpy as np
import os
import random
from typing import Dict, Tuple, TypeVar

class MCMCMethods(Enum):
    Metropolis = 'Metropolis method'
    Glauber = 'Glauber dynamics'
    SCA = 'Stochastic Cellular Automata'

class NodeType(Enum):
    Spin = frozenset({-1, +1})
    Binary = frozenset({0, 1})

NodeName = TypeVar('NodeName', int, str)

nodename_type = deferred_type()
nodename_type.define(NodeName.class_type.instance_type)
kv_ty_int = (NodeName, types.int8)
#kv_ty_float = (NodeName, types.float64)
#kv_ty_dict = (NodeName, types.DictType(*kv_ty_float))
spec = [
    ('__temperature', types.float64),
    ('__pinningParameter', types.float64),
    ('__spins', types.DictType(*kv_ty_int)),
    #('__externalMagneticFields', types.DictType(*kv_ty_float)),
    #('__couplingCoefficients', types.DictType(*kv_ty_dict)),
    ('__spins', types.DictType(*(typeof(NodeName), types.float64))),
    ('MarkovChain', MCMCMethods)
]

@jitclass(spec)
class IsingModel:
    """description of class"""
    def __init__(self, linear: Dict[NodeName, float], quadratic: Dict[Tuple[NodeName, NodeName], float]):
        self.__temperature: float = 0
        self.__pinningParameter: float = 0
        # HACK: 辞書型よりもリスト型を使った方が高速かもしれない。
        self.__spins: Dict[NodeName, int] = {
            node: 1
            for node in {n for pair in quadratic.keys() for n in pair}.union(linear.keys())
        }
        self.__externalMagneticFields: Dict[NodeName, float] = {
            node: linear[node] if node in linear else 0
            for node in self.__spins.keys()
        }
        #self.__couplingCoefficients: Dict[NodeName, Dict[NodeName, float]] = {
        #    row: {
        #        column: quadratic[(row, column)] if (row, column) in quadratic else 0
        #        for column in self.__spins.keys()
        #    }
        #    for row in self.__spins.keys()
        #}
        self.__couplingCoefficients: Dict[NodeName, Dict[NodeName, float]] = collections.defaultdict(dict)
        for row in self.__spins.keys():
            for column in self.__spins.keys():
                if row == column:  # 対角成分は強制的に0にする。
                    self.__couplingCoefficients[row][column] = 0
                else:
                    if (row, column) in quadratic:
                        self.__couplingCoefficients[row][column] = quadratic[(row, column)]
                    elif (column, row) in quadratic:
                        self.__couplingCoefficients[row][column] = quadratic[(column, row)]
                    else:
                        self.__couplingCoefficients[row][column] = 0
        self.MarkovChain: MCMCMethods = MCMCMethods.Metropolis

    def CalcLocalMagneticField(self, node: NodeName, spins: Dict[NodeName, int] = None) -> float:
        if not spins:
            spins = self.__spins
        return self.__externalMagneticFields[node] + sum([
            self.__couplingCoefficients[node][neighbor]
            for neighbor in spins.keys()
        ])

    @property
    def Spins(self) -> Dict[NodeName, int]:
        return self.__spins

    @property
    def ExternalMagneticField(self) -> Dict[NodeName, float]:
        return self.__externalMagneticFields

    @property
    def CouplingCoefficients(self) -> Dict[NodeName, Dict[NodeName, float]]:
        return self.__couplingCoefficients

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
        for index, spin in enumerate(self.__spins.values(), 1):
            print(0 if spin == -1 else 1, end='')
            if index % maxColumns == 0 or index == len(self.__spins):
                print()
            else:
                print(' ', end='')

    def Update(self):
        def metropolisMethod():
            updatedNode: NodeName = random.choice(list(self.__spins.keys()))  # 一旦、リストに変換するため遅い？
            energyDifference: float = 2.e0 * self.__spins[updatedNode] * self.CalcLocalMagneticField(updatedNode)
            if energyDifference < 0.e0:
                self.__spins[updatedNode] = -self.__spins[updatedNode]
            elif random.random() <= (math.exp(-energyDifference / self.__temperature) if self.__temperature != 0.e0 else 0.e0):
                self.__spins[updatedNode] = -self.__spins[updatedNode]

        def glauberDynamics():
            updatedNode: NodeName = random.choice(list(self.__spins.keys()))  # 一旦、リストに変換するため遅い？
            if random.random() <= 1.e0 / (1.e0 + math.exp(-2.e0 * self.CalcLocalMagneticField(updatedNode) / self.__temperature)):
                self.__spins[updatedNode] = +1
            else:
                self.__spins[updatedNode] = -1

        def sca():
            spins = self.__spins
            with mp.Pool(processes=16) as pool:
                self.__spins = dict(pool.map(functools.partial(UpdateOneSpinForSCA, spins=spins, isingModel=self), spins.keys()))

        if self.MarkovChain == MCMCMethods.Metropolis:
            metropolisMethod()
        elif self.MarkovChain == MCMCMethods.Glauber:
            glauberDynamics()
        elif self.MarkovChain == MCMCMethods.SCA:
            sca()
        else:
            raise ValueError('Illeagal choises')

    def GetEnergy(self):
        return 0.5e0 * sum([  # Remove double-counting duplicates by multiplying the sum by 1/2.
            -self.__spins[node] * self.CalcLocalMagneticField(node)
            for node in self.__spins.keys()
        ])

# 並列化の都合上、外部関数として定義する。
#@jit(parallel=True)
def UpdateOneSpinForSCA(node: NodeName, spins: Dict[NodeName, int], isingModel: IsingModel) -> Tuple[NodeName, int]:
    if random.random() <= 1.e0 / (1.e0 + math.exp((spins[node] * isingModel.CalcLocalMagneticField(node, spins) + isingModel.PinningParameter) / isingModel.Temperature)):
        return (node, -spins[node])
    else:
        return (node, spins[node])