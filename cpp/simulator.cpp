#include "simulator.h"
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

// quadraticのキーのペア (i, j) は順番が i < j となっていなければならない。
IsingModel::IsingModel(const LinearBiases linear, const QuadraticBiases quadratic)
{
	// spinsの添字と頂点の名前との対応表を作成。
	std::set<Node> nodes;
	for (auto iter = linear.begin(); iter != linear.end(); iter++)
		nodes.insert(iter->first);
	for (auto iter = quadratic.begin(); iter != quadratic.end(); iter++) {
		nodes.insert(iter->first.first);
		nodes.insert(iter->first.second);
	}
	nodeLabels = std::vector<Node>(nodes.begin(), nodes.end());

	// Hamiltonianの定数と変数を初期化。
	auto maxNodes = nodeLabels.size();
	spins.setConstant(maxNodes, Spin::Up);
	externalMagneticField.resize(maxNodes);
	for (auto i = 0; i < maxNodes; i++) {
		auto iter = linear.find(nodeLabels[i]);
		if (iter != linear.end())
			externalMagneticField(i) = iter->second;
		else
			externalMagneticField(i) = 0.e0;
	}
	couplingCoefficients = Eigen::MatrixXd::Zero(maxNodes, maxNodes);
	for (auto i = 0; i < maxNodes; i++) {
		for (auto j = i + 1; j < maxNodes; j++) {
			auto iter = quadratic.find(std::make_pair(nodeLabels[i], nodeLabels[j]));
			if (iter != quadratic.end())
				couplingCoefficients(i, j) = couplingCoefficients(j, i) = iter->second;
			else
				couplingCoefficients(i, j) = couplingCoefficients(j, i) = 0.e0;
		}
	}
}

void IsingModel::Print()
{
	std::cout << "Current spin configuration:" << std::endl;
	for (auto i = 0; i < spins.size(); i++)
		std::cout << std::setw(2) << spins(i);
	std::cout << std::endl;
	std::cout << "External magnetic field:" << std::endl;
	std::cout << externalMagneticField.transpose() << std::endl;
	std::cout << "Coupling coefficinets:" << std::endl;
	std::cout << couplingCoefficients << std::endl;
	std::cout << "Algorithm: " << AlgorithmToStr(algorithm) << std::endl;
}

/* Hamiltonian: H(s) = - sum_<x,y> J_{xy} s_x s_y - sum_x h_x s_x */
void IsingModel::Update()
{
	static auto MetropolisMethod = [this]() {
		unsigned int updatedNodeIndex = rand(spins.size());
		double energyDifference = 2.e0 * static_cast<int>(spins(updatedNodeIndex)) * calcLocalMagneticField(updatedNodeIndex);
		if (energyDifference < 0.e0)
			spins(updatedNodeIndex) = flip(spins(updatedNodeIndex));
		else if (rand.Bernoulli(std::exp(-energyDifference / temperature)))
			spins(updatedNodeIndex) = flip(spins(updatedNodeIndex));
	};

	static auto GlauberDynamics = [this]() {
		unsigned int updatedNodeIndex = rand(spins.size());
		if (rand.Bernoulli(1.e0 / (1.e0 + std::exp(-2.e0 * calcLocalMagneticField(updatedNodeIndex) / temperature))))
			spins(updatedNodeIndex) = Spin::Up;
		else
			spins(updatedNodeIndex) = Spin::Down;
	};

	static auto StochasticCellularAutomata = [this]() {
		// アルゴリズム部分。
		auto spins = this->spins;
		auto localMagneticField = calcLocalMagneticField(spins);
		auto updateOneSpinForRangeOf = [this, &spins, &localMagneticField](int begin, int end) {
			for (auto index = begin; index < end; index++) {
				if (rand.Bernoulli(1.e0 / (1.e0 + std::exp((static_cast<int>(spins(index)) * localMagneticField(index) + pinningParameter) / temperature))))
					this->spins(index) = flip(spins(index));
			}
		};

		// 並列処理部分。
		const int NumThreads = 32;
		std::vector<std::future<void>> tasks;
		tasks.reserve(NumThreads - 1);
		for (auto i = 0; i < NumThreads - 1; i++) {
			tasks.emplace_back(std::async(std::launch::async, updateOneSpinForRangeOf, i * spins.size() / NumThreads, (i + 1) * spins.size() / NumThreads));
		}
		updateOneSpinForRangeOf((NumThreads - 1) * spins.size() / NumThreads, spins.size());
	};

	static auto HillClimbing = [this]() {
		Configuration currentConfiguration = spins;
		while (true) {
			//double nextEval = std::numeric_limits<double>::infinity();
			double energyDifference = 0.e0;
			Configuration nextConfiguration = currentConfiguration;
			for (auto i = 0; i < spins.size(); i++) {
				double beforeEnergy = -1.e0 * static_cast<int>(currentConfiguration(i)) * calcLocalMagneticField(i);
				double afterEnergy = -1.e0 * static_cast<int>(flip(currentConfiguration(i))) * calcLocalMagneticField(i);
				if (energyDifference > afterEnergy - beforeEnergy) {
					energyDifference = afterEnergy - beforeEnergy;
					nextConfiguration(i) = flip(nextConfiguration(i));
				}
			}
			if (energyDifference >= 0.e0)
				break;
			currentConfiguration = nextConfiguration;
		}
		spins = currentConfiguration;
	};

	switch (algorithm) {
	case Algorithm::Metropolis:
		MetropolisMethod();
		break;
	case Algorithm::Glauber:
		GlauberDynamics();
		break;
	case Algorithm::SCA:
		StochasticCellularAutomata();
		break;
	case Algorithm::HillClimbing:
		HillClimbing();
		break;
	default:
		break;
	}
}
