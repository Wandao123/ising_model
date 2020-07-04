#include "simulator.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>

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
		.def_property_readonly("Spins", [](const IsingModel& self) -> std::map<Node, int> {
			std::map<Node, int> temp;
			for (auto pair : self.GetSpins())
				temp[pair.first] = pair.second;
			return temp;
		})
		.def("Write", [](const IsingModel& self) {
			py::scoped_ostream_redirect stream(
				std::cout,
				py::module::import("sys").attr("stdout")
			);
			self.Write();
		})
		.def("Update", &IsingModel::Update);
	py::enum_<IsingModel::Algorithms>(m, "Algorithms")
		.value("Metropolis", IsingModel::Algorithms::Metropolis)
		.value("Glauber", IsingModel::Algorithms::Glauber)
		.value("SCA", IsingModel::Algorithms::SCA)
		.value("HillClimbing", IsingModel::Algorithms::HillClimbing)
		.export_values();
}