#include "simulator.h"
#include <Eigen/Eigenvalues>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

// quadraticのキーのペア (i, j) は順番が i < j となっていなければならない。
IsingModel::IsingModel(const LinearBiases linear, const QuadraticBiases quadratic)
	: rand(std::make_unique<Rand>())
	, temperature(0.e0)
	, pinningParameter(0.e0)
	, algorithm(Algorithms::Metropolis)
{
	// spinsの添字と頂点の名前との対応表を作成。
	std::set<Node> nodes;
	for (auto iter = linear.begin(); iter != linear.end(); iter++)
		nodes.insert(iter->first);
	for (auto iter = quadratic.begin(); iter != quadratic.end(); iter++) {
		nodes.insert(iter->first.first);
		nodes.insert(iter->first.second);
	}
	std::size_t index = 0;
	for (const auto& key : nodes)
		nodeIndices[key] = index++;

	// Hamiltonianの定数と変数を初期化。
	auto maxNodes = nodeIndices.size();
	spins.setConstant(maxNodes, Spin::Up);
	previousSpins = spins;
	externalMagneticField.resize(maxNodes);
	for (const auto& node : nodeIndices) {
		auto iter = linear.find(node.first);
		if (iter != linear.end())
			externalMagneticField(node.second) = iter->second;
		else
			externalMagneticField(node.second) = 0.e0;
	}
	couplingCoefficients = Eigen::MatrixXd::Zero(maxNodes, maxNodes);
	for (const auto& row : nodeIndices) {
		for (const auto& column : nodeIndices) {
			if (row.first > column.first)
				continue;
			auto iter = quadratic.find(std::make_pair(row.first, column.first));
			if (iter != quadratic.end())
				couplingCoefficients(row.second, column.second) = couplingCoefficients(column.second, row.second) = iter->second;
			else
				couplingCoefficients(row.second, column.second) = couplingCoefficients(column.second, row.second) = 0.e0;
		}
	}
}

// 行列 (-J_{x, y})_{x, y} の最大固有値を計算する。
double IsingModel::CalcLargestEigenvalue() const
{
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(-couplingCoefficients);
	return solver.eigenvalues().reverse()(0);
}

double IsingModel::GetEnergy() const
{
	auto hamiltonian = [this]() -> double {
		// Remove double-counting duplicates by multiplying the sum by 1/2.
		return -spins.cast<double>().transpose() * (0.5e0 * couplingCoefficients * spins.cast<double>() + externalMagneticField);
	};

	auto hamiltonianOnBipartiteGraph = [this]() -> double {
		return -0.5e0 * spins.cast<double>().transpose() * couplingCoefficients * spins.cast<double>()
			- 0.5e0 * externalMagneticField.dot(spins.cast<double>() + previousSpins.cast<double>())
			+ 0.5e0 * pinningParameter * (spins.size() - spins.cast<double>().dot(previousSpins.cast<double>()));
	};

	switch (algorithm) {
	case Algorithms::Metropolis:
	case Algorithms::Glauber:
	case Algorithms::HillClimbing:
		return hamiltonian();
	case Algorithms::SCA:
	case Algorithms::MA:
	case Algorithms::MMA:
		return hamiltonianOnBipartiteGraph();
	default:
		return std::nan("");
		break;
	}
}

void IsingModel::GiveSpins(const ConfigurationsType configurationType)
{
	switch (configurationType) {
	case ConfigurationsType::AllDown:
		spins.fill(Spin::Down);
		break;
	case ConfigurationsType::AllUp:
		spins.fill(Spin::Up);
		break;
	case ConfigurationsType::Uniform:
		for (auto i = 0; i < spins.size(); i++)
			spins(i) = rand->Bernoulli(0.5e0) ? Spin::Down : Spin::Up;
		break;
	default:
		break;
	}
	previousSpins = spins;
}

void IsingModel::Update()
{
	auto metropolisMethod = [this]() {
		unsigned int updatedNodeIndex = (*rand)(spins.size());
		double energyDifference = 2.e0 * static_cast<int>(spins(updatedNodeIndex)) * calcLocalMagneticField(updatedNodeIndex);
		if (energyDifference < 0.e0)
			spins(updatedNodeIndex) = flip(spins(updatedNodeIndex));
		else if (rand->Bernoulli(std::exp(-energyDifference / temperature)))
			spins(updatedNodeIndex) = flip(spins(updatedNodeIndex));
	};

	auto glauberDynamics = [this]() {
		unsigned int updatedNodeIndex = (*rand)(spins.size());
		if (rand->Bernoulli(1.e0 / (1.e0 + std::exp(-2.e0 * calcLocalMagneticField(updatedNodeIndex) / temperature))))
			spins(updatedNodeIndex) = Spin::Up;
		else
			spins(updatedNodeIndex) = Spin::Down;
	};

	auto stochasticCellularAutomata = [this]() {
		previousSpins = spins;
		spins = (
			calcLocalMagneticField(spins) + pinningParameter * spins.cast<double>()
			- temperature * Eigen::VectorXd::NullaryExpr(spins.size(), [this]() -> double { return rand->Logistic(); })
		).array().sign().cast<Spin>();  // 実質起こらないが、符号関数に渡しているため、スピンが0になる場合がある。
	};

	// 温度を下げなければ ``annealing'' ではないが、論文では区別していないので、ここでもこの名称を用いる。
	auto momentumAnnealing = [this]() {
		Configuration temp = (
			calcLocalMagneticField(spins) + pinningParameter * spins.cast<double>()
			- temperature * Eigen::VectorXd::NullaryExpr(spins.size(), [this]() -> double { return rand->Exponential(); }).cwiseProduct(previousSpins.cast<double>())
		).array().sign().cast<Spin>();  // 実質起こらないが、符号関数に渡しているため、スピンが0になる場合がある。
		previousSpins = spins;
		spins = temp;
	};

	auto modifiedMomentumAnnealing = [this]() {
		previousSpins = spins;
		spins = (
			calcLocalMagneticField(spins) + pinningParameter * spins.cast<double>()
			- temperature * Eigen::VectorXd::NullaryExpr(spins.size(), [this]() -> double { return rand->Exponential(); }).cwiseProduct(spins.cast<double>())
		).array().sign().cast<Spin>();  // 実質起こらないが、符号関数に渡しているため、スピンが0になる場合がある。
	};

	auto hillClimbing = [this]() {
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
	case Algorithms::Metropolis:
		metropolisMethod();
		break;
	case Algorithms::Glauber:
		glauberDynamics();
		break;
	case Algorithms::SCA:
		stochasticCellularAutomata();
		break;
	case Algorithms::MA:
		momentumAnnealing();
		break;
	case Algorithms::MMA:
		modifiedMomentumAnnealing();
		break;
	case Algorithms::HillClimbing:
		hillClimbing();
		break;
	default:
		break;
	}
}

void IsingModel::Write() const
{
	std::cout << "Current spin configuration:" << std::endl;
	for (auto i = 0; i < spins.size(); i++)
		std::cout << std::setw(2) << static_cast<int>(spins(i));
	std::cout << "External magnetic field:" << std::endl;
	std::cout << externalMagneticField.transpose() << std::endl;
	std::cout << "Coupling coefficinets:" << std::endl;
	std::cout << couplingCoefficients << std::endl;
	std::cout << "Algorithm: " << AlgorithmToStr(algorithm) << std::endl;
	std::cout << "Temperature: " << temperature << std::endl;
	std::cout << "Pinning parameter: " << pinningParameter << std::endl;
}