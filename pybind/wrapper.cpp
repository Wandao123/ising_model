#include "../cpp/simulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

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
		.def_property_readonly("Spins", &IsingModel::GetSpins)
		.def("Write", &IsingModel::Write)
		.def("Update", &IsingModel::Update);
	py::enum_<IsingModel::Algorithms>(m, "Algorithms")
		.value("Metropolis", IsingModel::Algorithms::Metropolis)
		.value("Glauber", IsingModel::Algorithms::Glauber)
		.value("SCA", IsingModel::Algorithms::SCA)
		.value("HillClimbing", IsingModel::Algorithms::HillClimbing)
		.export_values();
}