#ifndef SIMULATOR_H
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

	std::uint_fast64_t operator()()
	{
		return mt();
	}

	std::int_fast64_t operator()(const std::int_fast64_t maximum)
	{
		std::uniform_int_distribution<> dist(0, (maximum >= 0) ? maximum - 1 : 0);
		return dist(mt);
	}

	std::int_fast64_t operator()(const std::int_fast64_t minimum, const std::int_fast64_t maximum)
	{
		std::uniform_int_distribution<> dist((minimum <= maximum) ? minimum : maximum, (minimum <= maximum) ? maximum : minimum);
		return dist(mt);
	}

	bool Bernoulli(const double probability)
	{
		std::bernoulli_distribution dist(probability);
		return dist(mt);
	}

	double Uniform()
	{
		std::uniform_real_distribution<double> dist(0.e0, 1.e0);
		return dist(mt);
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
	enum class Algorithm {
		Metropolis,
		Glauber,
		SCA,
		HillClimbing,
		SIZE
	};

	IsingModel(const LinearBiases linear, const QuadraticBiases quadratic);
	void Print();
	void Update();

	std::string AlgorithmToStr(Algorithm algorithm)
	{
		switch (algorithm) {
		case Algorithm::Metropolis:
			return { "Metropolis method" };
		case Algorithm::Glauber:
			return { "Glauber dynamics" };
		case Algorithm::SCA:
			return { "SCA" };
		case Algorithm::HillClimbing:
			return { "Hill climbing" };
		default:
			return {};
		}
	}

	void ChangeAlgorithmTo(const Algorithm algorithm)
	{
		this->algorithm = algorithm;
	}

	double GetEnergy()
	{
		// Remove double-counting duplicates by multiplying the sum by 1/2.
		return -spins.cast<double>().transpose() * (0.5e0 * couplingCoefficients * spins.cast<double>() + externalMagneticField);
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

	std::map<Node, Spin> GetSpins() const
	{
		std::map<Node, Spin> result;
		for (auto i = 0; i < nodeLabels.size(); i++)
			result[nodeLabels[i]] = spins[i];
		return result;
	}
private:
	using Configuration = Eigen::Matrix<Spin, Eigen::Dynamic, 1>;

	double temperature = 0.e0;        // Including the Boltzmann constant: k_B T.
	double pinningParameter = 0.e0;   // An parameter for the PCA.
	std::vector<Node> nodeLabels;
	Configuration spins;
	Eigen::VectorXd externalMagneticField;
	Eigen::MatrixXd couplingCoefficients;
	Algorithm algorithm = Algorithm::Metropolis;
	Rand rand;

	double calcLocalMagneticField(const unsigned int nodeIndex)
	{
		return (couplingCoefficients * spins.cast<double>())(nodeIndex) + externalMagneticField(nodeIndex);
	}

	Eigen::VectorXd calcLocalMagneticField(const Configuration& spins)
	{
		return couplingCoefficients * spins.cast<double>() + externalMagneticField;
	}

	Spin flip(Spin spin)
	{
		return (spin == Spin::Down) ? Spin::Up : Spin::Down;
	}
};

#endif // !SIMULATOR_H