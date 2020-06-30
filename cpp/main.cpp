#include "simulator.h"
//#include <chrono>
//#include <ctime>
#include <iomanip>
#include <iostream>
//#include <sstream>

int main()
{
    //const unsigned int maxNodes = 1024;
    //const double probability = 0.5e0;
    const unsigned int terminatedTime = static_cast<int>(1.e6);
    
    /*Rand rand;
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
    IsingModel isingModel({}, quadratic);*/
    IsingModel isingModel({ {0, 1.e0}, {1, 2.e0} }, { {{0, 1}, -3.e0} });

    /*auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-&m-&d %X");
    std::cout << ss.str() << std::endl;*/
    
    isingModel.ChangeAlgorithmTo(IsingModel::Algorithm::SCA);
    isingModel.SetTemperature(1.e0);
    //isingModel.SetPinningParameter(std::sqrt(maxNodes));
    isingModel.SetPinningParameter(3.e0 / 2);
    for (auto n = 0; n <= terminatedTime; n++) {
        isingModel.Update();
        std::cout << std::setw(7) << std::left << n
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
