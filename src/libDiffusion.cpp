#include <boost/multiprecision/float128.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "diffusionND.hpp"

typedef boost::multiprecision::float128 RealType;

namespace py = pybind11;

PYBIND11_MODULE(libDiffusion, m)
{
  py::class_<RandDistribution>(m, "RandDistribution")
      .def(py::init<const std::vector<double>>())
      .def("getAlpha", &RandDistribution::getAlpha)
      .def("getRandomNumbers", &RandDistribution::getRandomNumbers);
      
  py::class_<DiffusionND, RandDistribution>(m, "DiffusionND")
      .def(py::init<const std::vector<double>, unsigned int>())
      .def("getPDF", &DiffusionND::getPDF)
      .def("setPDF", &DiffusionND::setPDF)
      .def("iterateTimestep", &DiffusionND::iterateTimestep)
      .def("getRandomNumbers", &DiffusionND::getRandomNumbers)
      .def("getTime", &DiffusionND::getTime)
      .def("getL", &DiffusionND::getL)
      .def("integratedProbability", &DiffusionND::integratedProbability)
      .def("logIntegratedProbability", &DiffusionND::logIntegratedProbability);
}