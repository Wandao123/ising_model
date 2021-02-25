#include "simulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>
#include <optional>

namespace py = pybind11;

void Write(const Simulator::IsingModel& self)
{
	py::print("Current spin configuration:");
	py::print(self.GetSpins());
	py::print("External magnetic field:");
	py::print(self.GetExternalMagneticField());
	py::print("Coupling coefficinets:");
	py::print(self.GetCouplingCoefficients());
	py::print("Algorithm:", Simulator::AlgorithmToStr(self.GetCurrentAlgorithm()));
	py::print("Temperature:", self.GetTemperature());
	py::print("Pinning parameter:", self.GetPinningParameter());
	py::print("Flip trial rate:", self.GetFlipTrialRate());
}

PYBIND11_MODULE(simulatorWithCpp, m)
{
	m.doc() = "An Ising model simulator";
	m.def("AlgorithmToStr", &Simulator::AlgorithmToStr);
	py::bind_map<Simulator::LinearBiases>(m, "LinearBiases");
	py::bind_map<Simulator::QuadraticBiases>(m, "QuadraticBiases");
	py::class_<Simulator::IsingModel> isingModel(m, "IsingModel");
	isingModel.def(py::init<const Simulator::LinearBiases, const Simulator::QuadraticBiases>())
		.def_property("Algorithm", &Simulator::IsingModel::GetCurrentAlgorithm, &Simulator::IsingModel::ChangeAlgorithmTo)
		.def_property_readonly("Energy", &Simulator::IsingModel::GetEnergy)
		.def_property_readonly("EnergyOnBipartiteGraph", &Simulator::IsingModel::GetEnergyOnBipartiteGraph)
		.def_property("Temperature", &Simulator::IsingModel::GetTemperature, &Simulator::IsingModel::SetTemperature)
		.def_property("PinningParameter", &Simulator::IsingModel::GetPinningParameter, &Simulator::IsingModel::SetPinningParameter)
		.def_property("FlipTrialRate", &Simulator::IsingModel::GetFlipTrialRate, &Simulator::IsingModel::SetFlipTrialRate)
		.def_property("Spins",
			[](const Simulator::IsingModel& self) -> std::map<Simulator::Node, int> {
				std::map<Simulator::Node, int> temp;
				for (const auto& pair : self.GetSpinsAsDictionary())
					temp[pair.first] = static_cast<int>(pair.second);
				return temp;
			},
			[](Simulator::IsingModel& self, const std::map<Simulator::Node, int> spins) {
				std::map<Simulator::Node, Simulator::IsingModel::Spin> temp;
				for (const auto& pair : spins)
					if (pair.second == -1 || pair.second == +1)
						temp[pair.first] = static_cast<Simulator::IsingModel::Spin>(pair.second);
					else
						throw "Error: unable to convert " + std::to_string(pair.second);
				self.SetSpinsAsDictionary(temp);
			}
		)
		.def("CalcLargestEigenvalue", &Simulator::IsingModel::CalcLargestEigenvalue)
		.def("GiveSpins", &Simulator::IsingModel::GiveSpins)
		.def("SetSeed", [](Simulator::IsingModel& self, const std::optional<unsigned int> seed = std::nullopt) {
			if (seed)
				self.SetSeed(seed.value());
			else
				self.SetSeed();
		}, py::arg("seed") = std::nullopt)
		.def("Update", &Simulator::IsingModel::Update)
		.def("Write", &Write);
	py::enum_<Simulator::Algorithms>(m, "Algorithms")
		.value("Metropolis", Simulator::Algorithms::Metropolis)
		.value("Glauber", Simulator::Algorithms::Glauber)
		.value("SCA", Simulator::Algorithms::SCA)
		.value("fcSCA", Simulator::Algorithms::fcSCA)
		.value("MA", Simulator::Algorithms::MA)
		.value("MMA", Simulator::Algorithms::MMA)
		.value("HillClimbing", Simulator::Algorithms::HillClimbing)
		.export_values();
	py::enum_<Simulator::IsingModel::ConfigurationsType>(m, "ConfigurationsType")
		.value("AllDown", Simulator::IsingModel::ConfigurationsType::AllDown)
		.value("AllUp", Simulator::IsingModel::ConfigurationsType::AllUp)
		.value("Uniform", Simulator::IsingModel::ConfigurationsType::Uniform)
		.export_values();
}
