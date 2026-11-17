#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <Eigen/Geometry>

namespace py = pybind11;

namespace pybind11 {
namespace detail {

template <>
struct type_caster<Eigen::Isometry3d> {
    // set up conversion boilerplate
    PYBIND11_TYPE_CASTER(Eigen::Isometry3d, io_name("np.array", "np.array"));

    // C++ -> Python: convert `Eigen::Isometry3d` to `py::array`. Makes a copy
    // of the data
    static py::handle cast(
        const Eigen::Isometry3d& transform, return_value_policy, handle
    ) {
        constexpr ssize_t elem_size = sizeof(Eigen::Isometry3d::Scalar);
        return py::array(
                   py::dtype("float64"),
                   {transform.matrix().rows(), transform.matrix().cols()},
                   {elem_size * transform.matrix().rowStride(),
                    elem_size * transform.matrix().colStride()},
                   transform.data() /* Pointer to buffer */
        )
            .release();
    }

    // Python -> C++: convert a `py::array` into an `Eigen::Isometry3d` and
    // return false upon failure. The second argument indicates whether implicit
    // conversions should be allowed. The accepted types should reflect the type
    // hint specified by the first argument of `io_name`.
    bool load(handle src, bool /*convert*/) {
        // Check if handle is a Numpy array
        if (!py::isinstance<py::array>(src)) { return false; }
        auto array = py::reinterpret_borrow<py::array>(src);
        // Check if exactly two values are in the Sequence
        if (array.ndim() != 2 || array.shape(0) != 4 || array.shape(1) != 4) {
            return false;
        }
        if (!array.dtype().is(py::dtype("float64"))) { return false; }

        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                value.matrix()(i, j) =
                    *static_cast<const double*>(array.data(i, j));
            }
        }

        return true;
    }
};

template <>
struct type_caster<Eigen::Projective2d> {
    // set up conversion boilerplate
    PYBIND11_TYPE_CASTER(Eigen::Projective2d, io_name("np.array", "np.array"));

    // C++ -> Python: convert `Eigen::Projective2d` to `py::array`
    static py::handle cast(
        const Eigen::Projective2d& transform, return_value_policy, handle
    ) {
        return py::array(
                   py::dtype("float64"),
                   {3, 3},
                   {3, 8},
                   transform.data() /* Pointer to buffer */
        )
            .release();
    }

    // Python -> C++: convert a `py::array` into an `Eigen::Projective2d` and
    // return false upon failure. The second argument indicates whether implicit
    // conversions should be allowed. The accepted types should reflect the type
    // hint specified by the first argument of `io_name`.
    bool load(handle src, bool /*convert*/) {
        // Check if handle is a Numpy array
        if (!py::isinstance<py::array>(src)) { return false; }
        auto array = py::reinterpret_borrow<py::array>(src);
        // Check if exactly two values are in the Sequence
        if (array.ndim() != 2 || array.shape(0) != 3 || array.shape(1) != 3) {
            return false;
        }
        if (!array.dtype().is(py::dtype("float64"))) { return false; }

        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                value.matrix()(i, j) =
                    *static_cast<const double*>(array.data(i, j));
            }
        }

        return true;
    }
};

}  // namespace detail
}  // namespace pybind11
