#include "simulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/eigen.h>
#include <optional>

namespace py = pybind11;

void Write(const IsingModel& self)
{
	py::print("Current spin configuration:");
	py::print(self.GetSpins());
	py::print("External magnetic field:");
	py::print(self.GetExternalMagneticField());
	py::print("Coupling coefficinets:");
	py::print(self.GetCouplingCoefficients());
	py::print("Algorithm:", self.AlgorithmToStr(self.GetCurrentAlgorithm()));
	py::print("Temperature:", self.GetTemperature());
	py::print("Pinning parameter:", self.GetPinningParameter());
}

PYBIND11_MODULE(simulatorWithCpp, m)
{
	m.doc() = "An Ising model simulator";
	py::bind_map<LinearBiases>(m, "LinearBiases");
	py::bind_map<QuadraticBiases>(m, "QuadraticBiases");
	py::class_<IsingModel> isingModel(m, "IsingModel");
	isingModel.def(py::init<const LinearBiases, const QuadraticBiases>())
		.def_property("Algorithm", &IsingModel::GetCurrentAlgorithm, &IsingModel::ChangeAlgorithmTo)
		.def_property_readonly("Energy", &IsingModel::GetEnergy)
		.def_property("Temperature", &IsingModel::GetTemperature, &IsingModel::SetTemperature)
		.def_property("PinningParameter", &IsingModel::GetPinningParameter, &IsingModel::SetPinningParameter)
		.def_property_readonly("Spins", [](const IsingModel& self) -> std::map<Node, int> {
			std::map<Node, int> temp;
			for (auto pair : self.GetSpinsAsDictionary())
				temp[pair.first] = pair.second;
			return temp;
		})
		.def("CalcLargestEigenvalue", &IsingModel::CalcLargestEigenvalue)
		.def("GiveSpins", &IsingModel::GiveSpins)
		.def("SetSeed", [](IsingModel& self, const std::optional<unsigned int> seed = std::nullopt) {
			if (seed)
				self.SetSeed(seed.value());
			else
				self.SetSeed();
		}, py::arg("seed") = std::nullopt)
		.def("Update", &IsingModel::Update)
		.def("Write", &Write);
	py::enum_<IsingModel::Algorithms>(m, "Algorithms")
		.value("Metropolis", IsingModel::Algorithms::Metropolis)
		.value("Glauber", IsingModel::Algorithms::Glauber)
		.value("SCA", IsingModel::Algorithms::SCA)
		.value("MA", IsingModel::Algorithms::MA)
		.value("HillClimbing", IsingModel::Algorithms::HillClimbing)
		.export_values();
	py::enum_<IsingModel::ConfigurationsType>(m, "ConfigurationsType")
		.value("AllDown", IsingModel::ConfigurationsType::AllDown)
		.value("AllUp", IsingModel::ConfigurationsType::AllUp)
		.value("Uniform", IsingModel::ConfigurationsType::Uniform)
		.export_values();
}