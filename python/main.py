#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

import simulator
#import simulatorWithCpp as simulator

MaxTrials = int(1.e3)
MaxNodes = 64
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
    #quadratic = GenerateSquareLatticeEdges(MaxNodes)
    isingModel = simulator.IsingModel({}, quadratic)
    isingModel.Algorithm = simulator.Algorithms.fcSCA

    # Preprocess IsingModel's spin configuration and parameters.
    #isingModel.SetSeed(1024)  # Always use the same initial configuration.
    #isingModel.GiveSpins(simulator.ConfigurationsType.Uniform)
    spins = isingModel.Spins
    rng = np.random.default_rng(1024)
    for node in spins:
        spins[node] = rng.choice([-1, +1])
    isingModel.Spins = spins
    isingModel.SetSeed()
    if isingModel.Algorithm == simulator.Algorithms.SCA or isingModel.Algorithm == simulator.Algorithms.MA:
        isingModel.PinningParameter = isingModel.CalcLargestEigenvalue() / 2
    elif isingModel.Algorithm == simulator.Algorithms.fcSCA:
        isingModel.PinningParameter = isingModel.CalcLargestEigenvalue() / 8
        isingModel.FlipTrialRate = 0.8e0
    isingModel.Temperature = 2.e0 * np.sum([np.abs(J) for J in quadratic.values()]) + MaxNodes * isingModel.PinningParameter

    return isingModel

def SampleData(isingModel: simulator.IsingModel, initialTemperature: float) -> List[float]:
    result = np.empty((0, 3), dtype=np.float)
    for n in range(MaxTrials + 1):
        #isingModel.Temperature = initialTemperature / (np.sqrt(MaxNodes) * np.log(1 + n) + 1.e0)  # A logarithmic cooling schedule.
        #isingModel.Temperature = initialTemperature / (n + 1.e0)  # A linear multiplicative cooling schedule.
        #isingModel.Temperature = 1.e0 + (initialTemperature - 1.e0) * (MaxTrials - n) / MaxTrials  # A linear additive cooling schedule.
        isingModel.Temperature = initialTemperature * 0.99 ** n  # An exponential cooling schedule.
        #isingModel.Temperature = 1.e2 * np.exp(-1.e-2 * n)  # B. FK.'s results.
        #isingModel.Temperature = np.exp(-1.e-2 * (n - 200))
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
    isingModel.Write()
    print()
    data = SampleData(isingModel, isingModel.Temperature)
    DrawGraphFor(data)
    print('The minimum energy found by the simulator: {}'.format(np.amin(data[:, 1])))
    isingModel.Algorithm = simulator.Algorithms.Glauber
    print('Real energy:', isingModel.Energy)
