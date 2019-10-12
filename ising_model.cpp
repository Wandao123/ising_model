#include "ising_model.h"
#include <iostream>
#include <iomanip>
#include <random>
#include <future>
#include <limits>
#include <vector>

inline int Remainder(int Dividend, int Divisor)
{
	return (Dividend - Divisor * std::trunc(static_cast<double>(Dividend) / Divisor));
}

inline int Modulo(int Dividend, int Divisor)
{
	return (Dividend - Divisor * std::floor(static_cast<double>(Dividend) / Divisor));
}

IsingModel::IsingModel(double Temperature)
	: temperature(Temperature)
	, font(std::make_unique<FTPixmapFont>(FontFile.c_str()))
{
	if (font->Error()) {
		std::cerr << "Faild to open font: " << FontFile << std::endl;
		font.reset();
	} else {
		font->FaceSize(FontSize);
	}
	giveInitialConfiguration();
}

void IsingModel::Draw()
{
	//glEnableClientState(GL_VERTEX_ARRAY);
	glBegin(GL_QUADS);
	double tick = static_cast<double>(ScreenWidth) / SideLength;
	for (int i = 0; i < SideLength; i++) {
		for (int j = 0; j < SideLength; j++) {
			// Fills the box surrounded by (X1, Y1) and (X2, Y2)
			double X1 = j * tick, Y1 = i * tick;
			double X2 = (j + 1) * tick, Y2 = (i + 1) * tick;
			if (cells[i][j] == Status::UpSpin)
				glColor3d(1.0, 0.0, 0.0);
			else
				glColor3d(0.0, 0.0, 1.0);
			/*GLfloat vertexArray[] = {
				X1, Y1,
				X1, Y2,
				X2, Y2,
				X2, Y1,
			};
			glVertexPointer(2, GL_FLOAT, 0, vertexArray);
			glDrawArrays(GL_QUADS, 0, 4);*/
			glVertex2d(X1, Y1);
			glVertex2d(X1, Y2);
			glVertex2d(X2, Y2);
			glVertex2d(X2, Y1);
		}
	}
	glEnd();
	//glDisableClientState(GL_VERTEX_ARRAY);

	glColor3d(0.0, 0.0, 0.0);
	int posY = ScreenWidth;
	std::stringstream text;
	font->FaceSize(FontSize);
	text << "System size  = " << SideLength << " x " << SideLength << " = " << SideLength * SideLength;
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Temperature  = " << std::scientific << std::setprecision(5) << GetTemperature();
	if (algorithm == Algorithm::PCA)
		text << ", q = " << std::scientific << std::setprecision(5) << pinning;
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Energy       = " << std::scientific << std::setprecision(5) << GetEnergy();
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Auto cooling = " << (isCooling ? "ON" : "OFF");
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Algorithm    = " << AlgorithmToStr(algorithm);
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	font->FaceSize(FontSize * 2 / 3);
	text << "[sp] Start/Stop   [esc/q] Quit   [a] Cooling switch";
	drawText(text, font->LineHeight() / 2, posY += font->LineHeight() * 1.5);
	text << "[up/down] Inc./Dec. temperature   [c] Change algorithm";
	drawText(text, font->LineHeight() / 2, posY += font->LineHeight());
}

/* Hamiltonian: H(s) = - sum<i,j> J_{ij} s_i s_j */
void IsingModel::Update()
{
	static auto MetropolisMethod = [this]() {
		// アルゴリズム部分。
		auto updateOneSpin = [this]() {
			int X = makeRandomCoordinate();
			int Y = makeRandomCoordinate();
			double BeforeEnergy = -1.e0 * static_cast<int>(cells[Y][X]) * calcLocalMagneticField(X, Y);
			double AfterEnergy = -BeforeEnergy;
			double energyDifference = AfterEnergy - BeforeEnergy;
			if (energyDifference < 0.e0)
				cells[Y][X] = flip(cells[Y][X]);
			else if (giveRandomState(std::exp(-energyDifference / temperature)) == Status::UpSpin)
				cells[Y][X] = flip(cells[Y][X]);
		};

		// 繰り返し処理（本来は1回の更新につき1スピンのみだが、更新頻度をPCAに合わせる。
		for (int i = 1; i <= std::pow(SideLength, 2); i++)
			updateOneSpin();
	};

	static auto GlauberDynamics = [this]() {
		// アルゴリズム部分。
		auto updateOneSpin = [this]() {
			int X = makeRandomCoordinate();
			int Y = makeRandomCoordinate();
			if (giveRandomState(1.e0 / (1.e0 + std::exp(-2.e0 * calcLocalMagneticField(X, Y) / temperature))) == Status::UpSpin)
				cells[Y][X] = Status::UpSpin;
			else
				cells[Y][X] = Status::DownSpin;
		};

		// 繰り返し処理（本来は1回の更新につき1スピンのみだが、更新頻度をPCAに合わせる。
		for (int i = 1; i <= std::pow(SideLength, 2); i++)
			updateOneSpin();
	};

	static auto ProbabilisticCellularAutomata = [this]() {
		// アルゴリズム部分。
		const double V = std::pow(SideLength, 2);
		pinning = std::min(std::pow(std::log( 1.e0 + 1.e0 / temperature), 8.e0), 0.25e0 * std::log(V));
		//pinning = 1.e-4;
		// 論文の下界。
		//const double pinning = 0.5e0 * std::log(V) + V / temperature - 0.5e0 * std::log(1.e-4 * 0.25e0);
		// 独自解。
		//const double eps = 1.e-2;
		//const double phi = std::exp(4 * 1.e0 / temperature);
		//const double pinning = 0.5e0 * (std::log(phi - std::pow(phi, eps - 1)) - std::log(std::pow(phi, eps) - 1.e0));
		const std::array<std::array<Status, SideLength>, SideLength> cells = this->cells;
		auto updateOneSpinForRangeOf = [this, &cells](int begin, int end) {
			for (int X = begin; X < end; X++) {
				for (int Y = 0; Y < SideLength; Y++) {
					if (giveRandomState(1.e0 / (1.e0 + std::exp(2.e0 * static_cast<int>(cells[Y][X]) * calcLocalMagneticField(cells, X, Y) / temperature + 2.0e0 * pinning))) == Status::UpSpin)
						this->cells[Y][X] = flip(cells[Y][X]);
				}
			}
		};

		// 並列処理部分。
		const unsigned int NumThreads = 32;
		std::vector<std::future<void>> tasks;
		tasks.reserve(NumThreads - 1);
		for (auto i = 0; i < NumThreads - 1; i++) {
			tasks.emplace_back(std::async(std::launch::async, updateOneSpinForRangeOf, i * SideLength / NumThreads, (i + 1) * SideLength / NumThreads));
		}
		updateOneSpinForRangeOf((NumThreads - 1) * SideLength / NumThreads, SideLength);
	};

	static auto HillClimbing = [this]() {
		std::array<std::array<Status, SideLength>, SideLength> currentConfiguration = cells;
		while (true) {
			//double nextEval = std::numeric_limits<double>::infinity();
			double energyDifference = 0.e0;
			std::array<std::array<Status, SideLength>, SideLength> nextConfiguration = currentConfiguration;
			for (int X = 0; X < SideLength; X++) {
				for (int Y = 0; Y < SideLength; Y++) {
					double beforeEnergy = -1.e0 * static_cast<int>(currentConfiguration[Y][X]) * calcLocalMagneticField(X, Y);
					double afterEnergy = -1.e0 * static_cast<int>(flip(currentConfiguration[Y][X])) * calcLocalMagneticField(X, Y);
					if (energyDifference > afterEnergy - beforeEnergy) {
						energyDifference = afterEnergy - beforeEnergy;
						nextConfiguration[Y][X] = flip(nextConfiguration[Y][X]);
					}
				}
			}
			if (energyDifference >= 0.e0)
				break;
			currentConfiguration = nextConfiguration;
		}
		cells = currentConfiguration;
	};

	switch (algorithm) {
	case Algorithm::Metropolis:
		MetropolisMethod();
		break;
	case Algorithm::Glauber:
		GlauberDynamics();
		break;
	case Algorithm::PCA:
		ProbabilisticCellularAutomata();
		break;
	case Algorithm::HillClimbing:
		HillClimbing();
		break;
	}
	if (isCooling) {
		if (numSteps % CoolingInterval == 0)
			temperature = coolingSchedule(numSteps);
		++numSteps;
	}
}

double IsingModel::GetEnergy()
{
	double Result = 0.e0;
	for (int X = 0; X < SideLength; X++) {
		for (int Y = 0; Y < SideLength; Y++) {
			Result += -1.e0 * static_cast<int>(cells[Y][X]) * calcLocalMagneticField(X, Y);
		}
	}
	return (0.5e0 * Result);  // Remove double-counting duplicates
}

void IsingModel::ChangeAlgorithm()
{
	algorithm = static_cast<Algorithm>(Modulo(static_cast<int>(algorithm) + 1, static_cast<int>(Algorithm::SIZE)));
}

Status IsingModel::giveRandomState(double probability)
{
	std::random_device rand;
	std::mt19937 mt(rand());   // Mersenne twister
	std::uniform_real_distribution<double> spin(0.e0, 1.e0);
	return ((spin(mt) <= probability) ? Status::UpSpin : Status::DownSpin);
}

int IsingModel::makeRandomCoordinate()
{
	std::random_device rand;
	std::mt19937 mt(rand());
	std::uniform_int_distribution<int> Coordinate(0, SideLength - 1);
	return Coordinate(mt);
}

double IsingModel::getCouplingCoefficient(int iX, int iY, int jX, int jY)   // Nearest neighbor ferromagnet
{
	//int distance = std::abs(iX - jX) + std::abs(iY - jY);
	//return (distance == 1 ? +1.e0 : 0.e0);
	return +1;
}


void IsingModel::giveInitialConfiguration()
{
	for (int i = 0; i < SideLength; i++)
		for (int j = 0; j < SideLength; j++)
			cells[i][j] = (j < SideLength / 2) ? Status::UpSpin : Status::DownSpin;
}

double IsingModel::calcLocalMagneticField(const std::array<std::array<Status, SideLength>, SideLength>& cells, const int X, const int Y)
{
	int neighbor[4][2] = {   // Periodic boundary condition
			{Y, Modulo(X + 1, SideLength)},
			{Y, Modulo(X - 1, SideLength)},
			{Modulo(Y + 1, SideLength), X},
			{Modulo(Y - 1, SideLength), X}};
	double localMagneticField = 0.e0;
	for (int i = 0; i < 4; i++)
		localMagneticField += getCouplingCoefficient(X, Y, neighbor[i][1], neighbor[i][0])
			* static_cast<int>(cells[neighbor[i][0]][neighbor[i][1]]);
	return localMagneticField;
}

double IsingModel::calcLocalMagneticField(const int X, const int Y)
{
	return calcLocalMagneticField(this->cells, X, Y);
}

void IsingModel::drawText(std::stringstream& ss, const int posX, const int posY)
{
	glRasterPos2i(posX, posY);
	font->Render(ss.str().c_str());
	ss.str("");
}
