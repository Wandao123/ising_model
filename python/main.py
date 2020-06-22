#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import math
#import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Dict, Tuple

from simulators import IsingModel, MCMCMethods

# Erdo"s-Re'nyi random graph.
def generateErdosRenyiEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if random.random() <= 0.5e0 else 0 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# Spin glass.
def generateSpinGlassEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if random.random() <= 0.5e0 else 1 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

def printParameters(isingModel: IsingModel):
    print(dict(sorted(isingModel.Spins.items(), key=lambda x: x[0])))
    for i in range(maxNodes):
        for j in range(maxNodes):
            print('{0:2.0f}'.format(isingModel.CouplingCoefficients[i][j]), end='')
        print()
    print(isingModel.ExternalMagneticField)

if __name__ == '__main__':
    maxNodes = 1024
    isingModel = IsingModel({}, generateErdosRenyiEdges(maxNodes))
    isingModel.PinningParameter = math.sqrt(maxNodes) * 0.5e0
    initialTemperature = np.sum([np.abs(isingModel.CalcLocalMagneticField(isingModel.NodeIndices[node])) + isingModel.PinningParameter for node in isingModel.Spins.keys()])
    isingModel.Temperature = 200.e0
    #isingModel.MarkovChain = MCMCMethods.SCA

    output = []
    for i in range(2000):
        isingModel.Update()
        output.append([i, isingModel.GetEnergy(), isingModel.Temperature])
        isingModel.Temperature = 200.e0 - 0.1e0 * i
        if i % 20 == 0:
            print('Complete {0} times.  Energy={1}, Temperature={2}.'.format(i, output[i][1], output[i][2]))
    with open('output.dat', mode='w') as file:
        file.write('# date: ' + datetime.datetime.now().isoformat() + '.\n')
        for data in output:
            file.write('{0:<4d} {1:<14.5e} {2:<14.7e}\n'.format(data[0], data[1], data[2]))
