﻿#ifndef SIMULATOR_H
#define SIMULATOR_H

#include <Eigen/Core>
#include <cmath>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// Ref: https://qiita.com/Gaccho/items/dc312fb5a056505f0a9f
class Rand {
public:
	Rand()
	{
		mt.seed(rd());
	}

	void Seed()
	{
		mt.seed(rd());
	}

	void Seed(const std::int_fast64_t seed)
	{
		mt.seed(seed);
	}

	std::uint_fast64_t operator()()
	{
		return mt();
	}

	std::int_fast64_t operator()(const std::int_fast64_t maximum)
	{
		std::uniform_int_distribution<> distr(0, (maximum >= 0) ? maximum - 1 : 0);
		return distr(mt);
	}

	std::int_fast64_t operator()(const std::int_fast64_t minimum, const std::int_fast64_t maximum)
	{
		std::uniform_int_distribution<> distr((minimum <= maximum) ? minimum : maximum, (minimum <= maximum) ? maximum : minimum);
		return distr(mt);
	}

	bool Bernoulli(const double probability)
	{
		std::bernoulli_distribution distr(probability);
		return distr(mt);
	}

	double Exponential(const double intensity = 1.e0)
	{
		std::exponential_distribution<double> distr(intensity);
		return distr(mt);
	}

	double Logistic(const double location = 0.e0, const double scale = 1.e0)
	{
		double u = Uniform();
		return location + scale * std::log(u / (1.e0 - u));  // 逆関数法による生成。
	}

	double Uniform()
	{
		std::uniform_real_distribution<double> distr(0.e0, 1.e0);
		return distr(mt);
	}
private:
	std::random_device rd;
	std::mt19937_64 mt;
};

using Node = std::variant<int, std::string>;
using Edge = std::pair<Node, Node>;
using LinearBiases = std::map<Node, double>;
using QuadraticBiases = std::map<Edge, double>;

class IsingModel {
public:
	enum Spin : int {  // ライブラリ側でも型変換できるように、enum classではなくenumを使う。
		Down = -1,
		Up = +1
	};
	enum class Algorithms {
		Metropolis,
		Glauber,
		SCA,
		MA,
		HillClimbing,
		SIZE
	};
	enum class ConfigurationsType {
		AllDown,
		AllUp,
		Uniform
	};

	IsingModel(const LinearBiases linear, const QuadraticBiases quadratic);
	double CalcLargestEigenvalue() const;
	double GetEnergy() const;
	void GiveSpins(const ConfigurationsType configurationType);
	void Update();
	void Write() const;

	std::string AlgorithmToStr(Algorithms algorithm) const
	{
		switch (algorithm) {
		case Algorithms::Metropolis:
			return { "Metropolis method" };
		case Algorithms::Glauber:
			return { "Glauber dynamics" };
		case Algorithms::SCA:
			return { "Stochastic cellular automata" };
		case Algorithms::MA:
			return { "Momentum annealing" };
		case Algorithms::HillClimbing:
			return { "Hill climbing" };
		default:
			return { "Warning: Unknown type." };
		}
	}

	void SetSeed()
	{
		rand.Seed();
	}

	void SetSeed(const unsigned int seed)
	{
		rand.Seed(seed);
	}

	Algorithms GetCurrentAlgorithm() const
	{
		return algorithm;
	}

	void ChangeAlgorithmTo(const Algorithms algorithm)
	{
		this->algorithm = algorithm;
	}

	double GetTemperature() const
	{
		return temperature;
	}

	void SetTemperature(const double temperature)
	{
		this->temperature = (temperature >= 0.e0) ? temperature : 0.e0;
	}

	double GetPinningParameter() const
	{
		return pinningParameter;
	}

	void SetPinningParameter(const double pinningParameter)
	{
		this->pinningParameter = (pinningParameter >= 0.e0) ? pinningParameter : 0.e0;
	}

	std::map<Node, Spin> GetSpinsAsDictionary() const
	{
		std::map<Node, Spin> result;
		for (auto i = 0; i < nodeLabels.size(); i++)
			result[nodeLabels[i]] = spins[i];
		return result;
	}

	Eigen::VectorXi GetSpins() const
	{
		return spins.cast<int>();
	}

	Eigen::VectorXd GetExternalMagneticField() const
	{
		return externalMagneticField;
	}

	Eigen::MatrixXd GetCouplingCoefficients() const
	{
		return couplingCoefficients;
	}
private:
	using Configuration = Eigen::Matrix<Spin, Eigen::Dynamic, 1>;

	double temperature = 0.e0;        // Including the Boltzmann constant: k_B T.
	double pinningParameter = 0.e0;   // An parameter for the PCA.
	std::vector<Node> nodeLabels;
	Configuration spins;
	Configuration previousSpins;
	Eigen::VectorXd externalMagneticField;
	Eigen::MatrixXd couplingCoefficients;
	Algorithms algorithm = Algorithms::Metropolis;
	Rand rand;

	double calcLocalMagneticField(const unsigned int nodeIndex) const
	{
		return (couplingCoefficients * spins.cast<double>())(nodeIndex) + externalMagneticField(nodeIndex);
	}

	Eigen::VectorXd calcLocalMagneticField(const Configuration& spins) const
	{
		return couplingCoefficients * spins.cast<double>() + externalMagneticField;
	}

	Spin flip(const Spin spin) const
	{
		return (spin == Spin::Down) ? Spin::Up : Spin::Down;
	}
};

#endif // !SIMULATOR_H