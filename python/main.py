#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import math
#import matplotlib.pyplot as plt
#import numpy as np
import random

from simulators import IsingModel, MCMCMethods

if __name__ == '__main__':
    maxNodes = 64
    sideLength = math.ceil(math.sqrt(maxNodes))
    #couplings = []
    #for i in range(maxNodes):
    #    for j in {i - 1 if i % sideLength != 0 else i - 1 + sideLength, i + 1 if i % sideLength != sideLength - 1 else i + 1 - sideLength, i - sideLength if i >= sideLength else i + maxNodes - sideLength, i + sideLength if i < maxNodes - sideLength else i - maxNodes + sideLength}:
    #        couplings.append((i, j))
    #isingModel = IsingModel({}, {pair: 1 for pair in couplings})
    isingModel = IsingModel({}, {(i, j): -1 if random.random() <= 0.5e0 else 0 for i in range(maxNodes) for j in range(i + 1, maxNodes)})
    isingModel.PinningParameter = math.sqrt(maxNodes) * 0.5e0
    initialTemperature = sum([abs(isingModel.CalcLocalMagneticField(node)) + isingModel.PinningParameter for node in isingModel.Spins.keys()])
    isingModel.Temperature = 200.e0
    isingModel.MarkovChain = MCMCMethods.SCA

    output = []
    for i in range(2000):
        isingModel.Update()
        output.append([i, isingModel.GetEnergy(), isingModel.Temperature])
        isingModel.Temperature = 200.e0 - 0.1e0 * i
        if i % 20 == 0:
            print('Complete {0} times.  Energy={1}, Temperature={2}.'.format(i, output[i][1], output[i][2]))
    with open('output.dat', mode='w') as file:
        for data in output:
            file.write('{0:<4d} {1:<14.5e} {2:<14.7e}\n'.format(data[0], data[1], data[2]))

    """for i in range(1, 11):
        isingModel.Update()
        isingModel.Temperature = initialTemperature / math.log(i + 1)
        if i % 10 == 0:
            isingModel.Print()
            print(dict(sorted(isingModel.Spins.items(), key=lambda x: x[0])))
            print('Temperature={0:6.2f}, Energy={1:6.2f}'.format(isingModel.Temperature, isingModel.GetEnergy()))"""

    #for i in range(maxNodes):
    #    for j in range(maxNodes):
    #        print('{0:2d}'.format(isingModel.CouplingCoefficients[i][j]), end='')
    #    print()
    #print(isingModel.ExternalMagneticField)
