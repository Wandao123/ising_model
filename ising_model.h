#ifndef ISING_MODEL_H
#define ISING_MODEL_H

#include <array>
#include <memory>
#include <sstream>
#include <cmath>
#include <GLFW/glfw3.h>
#define FTGL_LIBRARY_STATIC
#include <FTGL/ftgl.h>

constexpr int ScreenWidth = 600;
constexpr int ScreenHeight = 800;
enum class Status {
	UpSpin = +1,
	DownSpin = -1
};

int Remainder(int Dividend, int Divisor);
int Modulo(int Dividend, int Divisor);

class IsingModel {
public:
	enum class Algorithm : int {
		Metropolis,
		Glauber,
		PCA,
		HillClimbing,
		SIZE
	};

	static constexpr unsigned int SideLength = 128;
	IsingModel(double Temperature);
	void Draw();
	void Update();
	double GetEnergy();
	void ChangeAlgorithm();

	void SwitchAutoCooling()
	{
		if (isCooling) {
			isCooling = false;
		} else {
			isCooling = true;
			numSteps = 0;
			initialTemperature = temperature;
		}
	}

	void Increase()
	{
		temperature += 1.e0 / NumDivision;
	}

	void Decrease()
	{
		temperature -= 1.e0 / NumDivision;
		if (temperature < 0.e0)
			temperature = 0.e0;
	}

	double GetTemperature()
	{
		return temperature;
	}
private:
#ifdef _WIN64
	const std::string FontFile = "C:/Windows/Fonts/consola.ttf";
#elif __linux__
	const std::string FontFile = "/usr/share/fonts/TTF/LiberationMono-Regular.ttf";
#endif
	const unsigned int FontSize = 22;
	const unsigned int NumDivision = 20;   // The variation of temperature
	const unsigned int CoolingInterval = static_cast<int>(std::pow(IsingModel::SideLength, 0));

	double initialTemperature = 0.e0;
	bool isCooling = false;
	unsigned long int numSteps = 0;
	Algorithm algorithm = Algorithm::Metropolis;
	double temperature;      // Include the Boltzmann constant: k_B T
	double pinning = 0.e0;   // An parameter for the PCA
	std::array<std::array<Status, SideLength>, SideLength> cells;
	std::unique_ptr<FTFont> font;
	Status giveRandomState(double probability);

	void giveInitialConfiguration();
	int makeRandomCoordinate();
	double getCouplingCoefficient(int iX, int iY, int jX, int jY);
	double calcLocalMagneticField(const std::array<std::array<Status, SideLength>, SideLength>& cells, const int X, const int Y);
	double calcLocalMagneticField(const int X, const int Y);
	void drawText(std::stringstream& ss, const int posX, const int posY);

	std::string AlgorithmToStr(Algorithm algorithm)
	{
		switch (algorithm) {
		case Algorithm::Metropolis:
			return { "Metropolis method" };
		case Algorithm::Glauber:
			return { "Glauber dynamics" };
		case Algorithm::PCA:
			return { "SCA" };
		case Algorithm::HillClimbing:
			return { "Hill climbing" };
		}
	}

	Status flip(Status state)
	{
		return ((state == Status::UpSpin) ? Status::DownSpin : Status::UpSpin);
	}

	double coolingSchedule(const int numTimes)
	{
		// Aarts, E.H.L. & Korst, J. (1989)
		static const double a = 2.e0;   // > 1
		return (initialTemperature / (1.e0 + a * std::log(1 + numTimes)));

		// Kirkpatrick, Gelatt and Vecchi (1983)
		//static const double a = 0.95e0;   // 0.8 <= a <= 0.9
		//return (initialTemperature * std::pow(a, numTimes));

		//static const double a = 0.0675;   // > 0
		//return (initialTemperature / (1.e0 + a * std::pow(numTimes, 2)));

		//static const double a = 9.45;   // > 0
		//return (initialTemperature / (1.e0 + a * numTimes));
	}
};

#endif // ! ISING_MODEL_H
