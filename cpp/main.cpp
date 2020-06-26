#include "simulator.h"
#include <iostream>

int main()
{
    const unsigned int maxNodes = 16;
    const double probability = 0.5e0;
    const unsigned int terminatedTime = 2000;
    
    Rand rand;
    QuadraticBiases quadratic;
    for (auto i = 0; i < maxNodes; i++) {
        for (auto j = i + 1; j < maxNodes; j++) {
            if (rand.Bernoulli(probability))
                quadratic[std::make_pair(i, j)] = -1.e0;
            else
                //quadratic[std::make_pair(i, j)] = 0.e0;
                quadratic[std::make_pair(i, j)] = +1.e0;
        }
    }
    IsingModel isingModel({}, quadratic);
    
    double T0 = std::pow(maxNodes, 2);
    double dT = (T0 - 1.e0) / terminatedTime;
    isingModel.ChangeAlgorithmTo(IsingModel::Algorithm::SCA);
    isingModel.SetPinningParameter(std::sqrt(maxNodes));
    for (auto n = 0; n <= terminatedTime; n++) {
        isingModel.SetTemperature(T0 - n * dT);
        isingModel.Update();
    }
    std::cout << "Energy = " << isingModel.GetEnergy() << std::endl;
    std::cout << "{ ";
    for (auto spin : isingModel.GetSpins())
        std::cout << "{" << std::get<int>(spin.first) << ": " << spin.second << "}, ";
    std::cout << "}\n" << std::endl;
    isingModel.Print();
    return 0;
}