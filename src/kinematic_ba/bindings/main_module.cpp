#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_slam_module(py::module&);

PYBIND11_MODULE(_core, m) {
    py::module_ slam_module = m.def_submodule("slam");
    init_slam_module(slam_module);
}
