#include "simulator.h"
//#include <chrono>
//#include <ctime>
#include <iomanip>
#include <iostream>
//#include <sstream>

int main()
{
    const unsigned int maxNodes = 1024;
    const double probability = 0.5e0;
    const unsigned int terminatedTime = 2000;
    
    Rand rand;
    QuadraticBiases quadratic;
    for (auto i = 0; i < maxNodes; i++) {
        for (auto j = i + 1; j < maxNodes; j++) {
            if (rand.Bernoulli(probability))
                quadratic[std::make_pair(i, j)] = -1.e0;
            else
                quadratic[std::make_pair(i, j)] = 0.e0;
                //quadratic[std::make_pair(i, j)] = +1.e0;
        }
    }
    IsingModel isingModel({}, quadratic);

    /*auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-&m-&d %X");
    std::cout << ss.str() << std::endl;*/
    
    //double T0 = std::pow(maxNodes, 2);
    double T0 = 200.e0;
    double dT = (T0 - 1.e0) / terminatedTime;
    isingModel.ChangeAlgorithmTo(IsingModel::Algorithm::SCA);
    //isingModel.SetPinningParameter(std::sqrt(maxNodes));
    isingModel.SetPinningParameter(22.4e0);
    for (auto n = 0; n <= terminatedTime; n++) {
        isingModel.SetTemperature(T0 - n * dT);
        isingModel.Update();
        std::cout << std::setw(4) << n
            << std::setw(16) << std::scientific << std::setprecision(5) << isingModel.GetEnergy()
            << std::setw(16) << std::scientific << std::setprecision(7) << isingModel.GetTemperature()
            << std::endl;
    }
    /*std::cout << "Energy = " << isingModel.GetEnergy() << std::endl;
    std::cout << "{ ";
    for (auto spin : isingModel.GetSpins())
        std::cout << "{" << std::get<int>(spin.first) << ": " << spin.second << "}, ";
    std::cout << "}\n" << std::endl;
    isingModel.Print();*/
    return 0;
}
