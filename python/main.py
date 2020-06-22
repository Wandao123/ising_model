#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import math
import matplotlib.pyplot as plt
import numpy as np

from simulators import IsingModel, MCMCMethods

if __name__ == '__main__':
    maxNodes = 256
    sideLength = math.ceil(np.sqrt(maxNodes))
    couplings = []
    for i in range(maxNodes):
        #for j in range(i + 1, maxNodes):
        for j in {i - 1 if i % sideLength != 0 else i - 1 + sideLength, i + 1 if i % sideLength != sideLength - 1 else i + 1 - sideLength, i - sideLength if i >= sideLength else i + maxNodes - sideLength, i + sideLength if i < maxNodes - sideLength else i - maxNodes + sideLength}:
            couplings.append((i, j))
    isingModel = IsingModel({}, {pair: 1 for pair in couplings})
    isingModel.PinningParameter = np.sqrt(maxNodes) * 0.5e0
    initialTemperature = np.sum(np.array([np.abs(isingModel.CalcLocalMagneticField(node)) + isingModel.PinningParameter for node in isingModel.Spins.keys()], dtype=np.float))
    isingModel.Temperature = initialTemperature
    isingModel.MarkovChain = MCMCMethods.SCA

    time = np.arange(1, 1001)
    energies = np.array([], dtype=np.float)
    for i in time:
        isingModel.Update()
        isingModel.Temperature = initialTemperature / np.log(i + 1)
        np.append(energies, isingModel.GetEnergy())
        #if i % 10 == 0:
        #    isingModel.Print()
        #    print(dict(sorted(isingModel.Spins.items(), key=lambda x: x[0])))
        #    print('Temperature={0:6.2f}, Energy={1:6.2f}'.format(isingModel.Temperature, isingModel.GetEnergy()))
    plt.plot(time, energies)
    plt.show()

    #for i in range(maxNodes):
    #    for j in range(maxNodes):
    #        print(isingModel.CouplingCoefficients[i][j], end='')
    #    print()
    #print(isingModel.ExternalMagneticField)