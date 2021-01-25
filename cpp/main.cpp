#include "simulator.h"
//#include <chrono>
#include <cmath>
//#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
//#include <sstream>

const unsigned int Seed = 32;

QuadraticBiases generateErdosRenyiEdges(const int maxNodes, const double probability)
{
    Rand rand;
    rand.Seed(Seed);
    QuadraticBiases result;
    for (auto i = 0; i < maxNodes; i++) {
        for (auto j = i + 1; j < maxNodes; j++) {
            if (rand.Bernoulli(probability))
                result[std::make_pair(i, j)] = -1.e0;
            else
                result[std::make_pair(i, j)] = 0.e0;
        }
    }
    return result;
}

QuadraticBiases generateSpinGlassEdges(const int maxNodes, const double probability)
{
    Rand rand;
    rand.Seed(Seed);
    QuadraticBiases result;
    for (auto i = 0; i < maxNodes; i++) {
        for (auto j = i + 1; j < maxNodes; j++) {
            if (rand.Bernoulli(probability))
                result[std::make_pair(i, j)] = -1.e0;
            else
                result[std::make_pair(i, j)] = +1.e0;
        }
    }
    return result;
}

void printStatus(const IsingModel& isingModel)
{
    std::cout << "Energy = " << isingModel.GetEnergy() << std::endl;
    std::cout << "{ ";
    for (auto spin : isingModel.GetSpinsAsDictionary())
        std::cout << "{" << std::get<int>(spin.first) << ": " << static_cast<int>(spin.second) << "}, ";
    std::cout << "}\n" << std::endl;
    isingModel.Write();
}

int main()
{
    const unsigned int maxNodes = 256;
    const double probability = 0.5e0;
    const unsigned int maxTrials = static_cast<int>(1.e4);
    auto quadratic = generateErdosRenyiEdges(maxNodes, probability);
    IsingModel isingModel({}, quadratic);
    isingModel.ChangeAlgorithmTo(IsingModel::Algorithms::fcSCA);
    switch (isingModel.GetCurrentAlgorithm()) {
    case IsingModel::Algorithms::SCA:
    case IsingModel::Algorithms::MA:
        isingModel.SetPinningParameter(isingModel.CalcLargestEigenvalue() * 0.5e0);
        break;
    case IsingModel::Algorithms::fcSCA:
        isingModel.SetFlipTrialRate(0.75e0);
        break;
    }
    double initialTemperature = std::accumulate(
        quadratic.begin(), quadratic.end(), 0.e0,
        [](double acc, const QuadraticBiases::value_type& p) -> double { return acc + std::abs(p.second); }
    ) + isingModel.GetPinningParameter();
    isingModel.SetTemperature(initialTemperature);
    //isingModel.SetSeed(Seed * 2);
    //isingModel.GiveSpins(IsingModel::ConfigurationsType::Uniform);
    auto spins = isingModel.GetSpinsAsDictionary();
    Rand rand;
    rand.Seed(Seed * 2);
    for (auto& spin : spins)
        spin.second = rand.Choice(std::vector<IsingModel::Spin>{ IsingModel::Spin::Down, IsingModel::Spin::Up });
    isingModel.SetSpinsAsDictionary(spins);
    isingModel.SetSeed();

    /*auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-&m-&d %X");
    std::cout << ss.str() << std::endl;*/

    isingModel.Write();
    for (auto n = 0; n <= maxTrials; n++) {
        //isingModel.SetTemperature(initialTemperature / (std::sqrt(maxNodes) * std::log(n + 1) + 1.e0));  // Alogarithmic cooling schedule.
        //isingModel.SetTemperature(initialTemperature / (n + 1.e0));  // A linear multiplicative cooling schedule.
        //isingModel.SetTemperature(1.e0 + (initialTemperature - 1.e0) * (maxTrials - n) / maxTrials);  // A linearadditive cooling schedule (whose final temperature is 1.e0).
        isingModel.SetTemperature(initialTemperature * std::pow(0.99e0, n));  // An exponential cooling schedule.
        isingModel.Update();
        std::cout << std::setw(7) << std::left << n
            << std::setw(16) << std::scientific << std::setprecision(5) << isingModel.GetEnergy()
            << std::setw(16) << std::scientific << std::setprecision(7) << isingModel.GetTemperature()
            << std::endl;
    }
    return 0;
}
