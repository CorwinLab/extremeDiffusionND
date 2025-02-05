#include <boost/multiprecision/float128.hpp>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pybind11_numpy_scalar.h"
#include "diffusionND.hpp"

typedef boost::multiprecision::float128 RealType;

namespace py = pybind11;

namespace pybind11 {
namespace detail {

// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 256;

// Kinda following:
// https://github.com/pybind/pybind11/blob/9bb3313162c0b856125e481ceece9d8faa567716/include/pybind11/numpy.h#L1000
template <> struct npy_format_descriptor<RealType> {
  static constexpr auto name = _("RealType");
  static pybind11::dtype dtype()
  {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
};

template <> struct type_caster<RealType> : npy_scalar_caster<RealType> {
  static constexpr auto name = _("RealType");
};

} // namespace detail
} // namespace pybind11

PYBIND11_MODULE(libDiffusion, m)
{
  py::class_<RandDistribution>(m, "RandDistribution")
      .def(py::init<const std::vector<double>>())
      .def("getAlpha", &RandDistribution::getAlpha)
      .def("setAlpha", &RandDistribution::setAlpha)
      .def("getRandomNumbers", &RandDistribution::getRandomNumbers);
      
  py::class_<DiffusionND, RandDistribution>(m, "DiffusionND")
      .def(py::init<const std::vector<double>, unsigned int>())
      .def("getPDF", &DiffusionND::getPDF)
      .def("setPDF", &DiffusionND::setPDF)
      .def("iterateTimestep", &DiffusionND::iterateTimestep)
      .def("getRandomNumbers", &DiffusionND::getRandomNumbers)
      .def("getTime", &DiffusionND::getTime)
      .def("setTime", &DiffusionND::setTime)
      .def("getL", &DiffusionND::getL)
      .def("integratedProbability", &DiffusionND::integratedProbability)
      .def("logIntegratedProbability", &DiffusionND::logIntegratedProbability);
}