#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wandao123'
__date__ = '2020/6/21'

from simulators import IsingModel, MCMCMethods

if __name__ == '__main__':
    isingModel = IsingModel({'a': 0}, {('a', 'b'): 1})
    isingModel.Temperature = 2.e0
    isingModel.MarkovChain = MCMCMethods.SCA
    isingModel.Update()
    #isingModel.Print()
    print(isingModel.Spins)
    print(isingModel.GetEnergy())