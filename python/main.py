#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Tuple

#import simulator
import simulatorWithCpp as simulator

# Erdo"s-Re'nyi random graph.
def generateErdosRenyiEdges(maxNodes: int, probability: float) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if np.random.random() <= probability else 0 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# Spin glass.
def generateSpinGlassEdges(maxNodes: int, probability: float) -> Dict[Tuple[int, int], float]:
    return {(i, j): -1 if np.random.random() <= probability else 1 for i in range(maxNodes) for j in range(i + 1, maxNodes)}

# 2-dimensional square lattice.
def generateSquareLatticeEdges(maxNodes: int) -> Dict[Tuple[int, int], float]:
    columns = math.ceil(math.sqrt(maxNodes))
    result = {}
    for i in range(maxNodes - 1):
        if (i + 1) % columns > 0:
            result[(i, i + 1)] = 1
        if (i + columns) < maxNodes:
            result[(i, i + columns)] = 1
    return result

if __name__ == '__main__':
    # ランダム・グラフ用の乱数の初期化。
    seed = 32
    np.random.seed(seed=seed)

    # 初期化。
    maxTrials = int(1.e5 + 1)
    maxNodes = 1024
    probability = 0.5e0
    quadratic = generateErdosRenyiEdges(maxNodes, probability)
    isingModel = simulator.IsingModel({}, quadratic)
    isingModel.PinningParameter = isingModel.CalcLargestEigenvalue() / 2
    isingModel.Algorithm = simulator.Algorithms.SCA

    # 前処理。
    if isingModel.Algorithm == simulator.Algorithms.SCA:
        isingModel.Update()
        initialTemperature = 2.e0 * np.sum([np.abs(J) for J in quadratic.values()]) + maxNodes * isingModel.PinningParameter
    else:
        for n in range(maxNodes):
            isingModel.Update()
        initialTemperature = 2.e0 * np.sum([np.abs(J) for J in quadratic.values()])
    print(np.array(list(isingModel.Spins.values())))

    # サンプリング。
    samples = np.empty((0, 3), dtype=np.float)
    for n in range(maxTrials):
        #isingModel.Temperature = initialTemperature / (np.sqrt(maxNodes) * np.log(1 + n) + 1.e0)  # 対数スケジュール。
        isingModel.Temperature = initialTemperature / (n + 1.e0)  # 線形積算スケジュール。
        #isingModel.Temperature = 1.e0 + (initialTemperature - 1.e0) * (maxTrials - n) / maxTrials  # 線形加算スケジュール。
        #isingModel.Temperature = initialTemperature * 0.98 ** n  # 指数スケジュール。
        isingModel.Update()
        samples = np.append(samples, np.array([n, isingModel.Energy, isingModel.Temperature], dtype=np.float).reshape((1, 3)), axis=0)
        if n % (maxTrials // 10) == 0:
            print('Complete {0} times.  Energy={1}, Temperature={2}.'.format(samples[n][0], samples[n][1], samples[n][2]))
    if samples.size > 0:
        np.savetxt(
            'output.dat',
            samples,
            fmt='%d %.5e %.7e',
            delimiter=' ',
            header='date: ' + datetime.datetime.now().isoformat() + '\n' + 'pinning parameter = ' + str(isingModel.PinningParameter) + '\n' + 'random seed = ' + str(seed)
        )

    # スコア。
    print('The minimum energy found by the simulator: {}'.format(np.amin(samples[:, 1])))

    # グラフの描画。
    fig = plt.figure(figsize=(4, 4), dpi=200)
    ax = fig.add_subplot(111)
    ax.plot(samples[:, 0], samples[:, 1])
    plt.show()
