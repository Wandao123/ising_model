#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import numpy as np
import random
from typing import Dict, Tuple

#import simulator
import simulatorWithCpp as simulator

# Erdo"s-Re'nyi random graph.
def generateErdosRenyiEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if random.random() <= 0.5e0 else 0 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# Spin glass.
def generateSpinGlassEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if random.random() <= 0.5e0 else 1 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

if __name__ == '__main__':
    maxNodes = 1024
    quadratic = generateErdosRenyiEdges(maxNodes)
    isingModel = simulator.IsingModel({}, quadratic)
    isingModel.PinningParameter = np.sqrt(maxNodes) / 2
    initialTemperature = np.sum([np.abs(J) + isingModel.PinningParameter for J in quadratic.values()])
    isingModel.Algorithm = simulator.Algorithms.SCA

    output = []
    for i in range(int(2.e3 + 1)):
        isingModel.Temperature = initialTemperature / (np.log(1 + i) + 1.e0)
        isingModel.Update()
        output.append([i, isingModel.Energy, isingModel.Temperature])
        if i % 20 == 0:
            print('Complete {0} times.  Energy={1}, Temperature={2}.'.format(i, output[i][1], output[i][2]))
    with open('output.dat', mode='w') as file:
        file.write('# date: ' + datetime.datetime.now().isoformat() + '.\n')
        for data in output:
            file.write('{0:<4d} {1:<14.5e} {2:<14.7e}\n'.format(data[0], data[1], data[2]))