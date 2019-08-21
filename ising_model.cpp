#include "ising_model.h"
#include <iostream>
#include <random>
#include <future>
#include <vector>

inline int Remainder(int Dividend, int Divisor)
{
	return (Dividend - Divisor * std::trunc(static_cast<double>(Dividend) / Divisor));
}

inline int Modulo(int Dividend, int Divisor)
{
	return (Dividend - Divisor * std::floor(static_cast<double>(Dividend) / Divisor));
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

void IsingModel::drawText(std::stringstream& ss, const int posX, const int posY)
{
	glRasterPos2i(posX, posY);
	font->Render(ss.str().c_str());
	ss.str("");
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
	text << "System size  = " << SideLength << " x " << SideLength << " = " << SideLength * SideLength;
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Temperature  = " << GetTemperature();
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Energy       = " << GetEnergy();
	drawText(text, font->LineHeight(), posY += font->LineHeight());
	text << "Auto cooling = " << (isCooling ? "ON" : "OFF");
	drawText(text, font->LineHeight(), posY += font->LineHeight());
}

/* Hamiltonian: H(s) = - sum<i,j> J_{ij} s_i s_j */
void IsingModel::Update()
{
	auto MetropolisAlgorithm = [this]() {
		int X = makeRandomCoordinate();
		int Y = makeRandomCoordinate();
		double BeforeEnergy = -1.e0 * static_cast<int>(cells[Y][X]) * calcLocalMagneticField(X, Y);
		double AfterEnergy = -BeforeEnergy;
		double energyDifference = AfterEnergy - BeforeEnergy;
		if (energyDifference < 0.e0) {
			cells[Y][X] = flip(cells[Y][X]);
		} else if (giveRandomState(std::exp(-energyDifference / temperature)) == Status::UpSpin) {
			cells[Y][X] = flip(cells[Y][X]);
		}
	};
	auto GlauberDynamics = [this]() {
		int X = makeRandomCoordinate();
		int Y = makeRandomCoordinate();
		if (giveRandomState(1.e0 / (1.e0 + std::exp(-2.e0 * calcLocalMagneticField(X, Y) / temperature))) == Status::UpSpin) {
			cells[Y][X] = Status::UpSpin;
		} else {
			cells[Y][X] = Status::DownSpin;
		}
	};
	auto ProbabilisticCellularAutomata = [this]() {
		const double q = 1.e0 / temperature;
		auto updateOneSpinForRangeOf = [this, q](int begin, int end) {
			for (int X = begin; X < end; X++) {
				for (int Y = 0; Y < SideLength; Y++) {
					if (giveRandomState(1.e0 / (1.e0 + std::exp(2.e0 * static_cast<int>(cells[Y][X]) * calcLocalMagneticField(X, Y) / temperature + 2.0e0 * q))) == Status::UpSpin)
						cells[Y][X] = flip(cells[Y][X]);
				}
			}
		};
		const unsigned int NumThreads = 20;
		std::vector<std::future<void>> tasks;
		tasks.reserve(NumThreads - 1);
		for (auto i = 0; i < NumThreads - 1; i++) {
			tasks.emplace_back(std::async(std::launch::async, updateOneSpinForRangeOf, i * SideLength / NumThreads, (i + 1) * SideLength / NumThreads));
		}
		updateOneSpinForRangeOf((NumThreads - 1) * SideLength / NumThreads, SideLength);
	};

	//MetropolisAlgorithm();
	GlauberDynamics();
	//ProbabilisticCellularAutomata();
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

double IsingModel::calcLocalMagneticField(const int X, const int Y)
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
