#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

#import simulator
import simulatorWithCpp as simulator

MaxTrials = int(5.e4)
MaxNodes = 256
Probability = 0.5e0

# Erdo"s-Re'nyi random graph.
def GenerateErdosRenyiEdges(maxNodes: int, probability: float) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if np.random.random() <= probability else 0 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# Spin glass.
def GenerateSpinGlassEdges(maxNodes: int, probability: float) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if np.random.random() <= probability else 1 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# 2-dimensional square lattice.
def GenerateSquareLatticeEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    columns = math.ceil(math.sqrt(maxNodes))
    result = {}
    for i in range(maxNodes - 1):
        if (i + 1) % columns > 0:
            result[(i, i + 1)] = -1
        if (i + columns) < maxNodes:
            result[(i, i + columns)] = -1
    return result

def Initialize() -> simulator.IsingModel:
    # Create a simulator.IsingModel instance.
    quadratic = GenerateErdosRenyiEdges(MaxNodes, Probability)
    isingModel = simulator.IsingModel({}, quadratic)
    isingModel.PinningParameter = isingModel.CalcLargestEigenvalue() / 2
    isingModel.Algorithm = simulator.Algorithms.SCA

    # Preprocess IsingModel's spin configuration.
    if isingModel.Algorithm == simulator.Algorithms.SCA:
        isingModel.Temperature = 2.e0 * np.sum([np.abs(J) for J in quadratic.values()]) + MaxNodes * isingModel.PinningParameter
        isingModel.Update()
    else:
        isingModel.Temperature = 2.e0 * np.sum([np.abs(J) for J in quadratic.values()])
        for n in range(MaxNodes):
            isingModel.Update()

    return isingModel

def SampleData(isingModel: simulator.IsingModel, initialTemperature: float) -> List[float]:
    result = np.empty((0, 3), dtype=np.float)
    for n in range(MaxTrials + 1):
        #isingModel.Temperature = initialTemperature / (np.sqrt(MaxNodes) * np.log(1 + n) + 1.e0)  # A logarithmic cooling schedule.
        isingModel.Temperature = initialTemperature / (n + 1.e0)  # A linear multiplicative cooling schedule.
        #isingModel.Temperature = 1.e0 + (initialTemperature - 1.e0) * (MaxTrials - n) / MaxTrials  # A linear additive cooling schedule.
        #isingModel.Temperature = initialTemperature * 0.99 ** n  # An exponential cooling schedule.
        isingModel.Update()
        result = np.append(result, np.array([n, isingModel.Energy, isingModel.Temperature], dtype=np.float).reshape((1, 3)), axis=0)
        if n % (MaxTrials // 10) == 0:
            print('Complete {0} times.  Energy={1}, Temperature={2}.'.format(result[n][0], result[n][1], result[n][2]))
    if result.size > 0:
        np.savetxt(
            'output.dat',
            result,
            fmt='%d %.5e %.7e',
            delimiter=' ',
            header='date: ' + datetime.datetime.now().isoformat() + '\n' + 'pinning parameter = ' + str(isingModel.PinningParameter) + '\n' + 'random seed = ' + str(seed)
        )
    return result

def DrawGraphFor(data: List[float]):
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(data[:, 0], data[:, 1])
    plt.show()

if __name__ == '__main__':
    # Fix random seed in Numpy.
    seed = 32
    np.random.seed(seed=seed)

    # Run a simulation.
    isingModel = Initialize()
    print(np.array(list(isingModel.Spins.values())))
    data = SampleData(isingModel, isingModel.Temperature)
    print('The minimum energy found by the simulator: {}'.format(np.amin(data[:, 1])))
    DrawGraphFor(data)
