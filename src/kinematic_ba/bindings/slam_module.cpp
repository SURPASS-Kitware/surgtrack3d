#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "eigen_geometry.hpp"
#include "kinematic_ba/slam/camera_model.hpp"
#include "kinematic_ba/slam/frame.hpp"
#include "kinematic_ba/slam/slam.hpp"

namespace py = pybind11;
using namespace kinematic_ba::slam;

void init_slam_module(py::module& m) {
    py::classh<CameraModel>(
        m, "CameraModel", "Rectified camera projection model"
    )
        .def(
            py::init<const Eigen::Isometry3d&, const Eigen::Projective2d&>(),
            R"(Construct from camera pose (optical frame wrt camera frame) and
               (rectified) camera intrinsic matrix. Optical frame may be different
               from camera frame for e.g. stereo cameras, where both camera heads
               share a single camera reference frame.)",
            py::arg("pose"),
            py::arg("intrinsic")
        )
        .def(
            py::init<const Eigen::Matrix<double, 3, 4>&>(),
            "Construct directly from camera projection matrix",
            py::arg("projection")
        )
        .def("projectionMatrix", &CameraModel::projectionMatrix);

    py::classh<Feature>(m, "Feature", "Tracked/unique feature point for SLAM")
        .def(
            py::init<const Eigen::Vector3d&>(),
            "Create new Feature given initial guess of 3D position",
            py::arg("position")
        )
        .def_readonly(
            "position",
            &Feature::position,
            R"(Current 3D estimate of feature position in world/base base)"
        );

    py::classh<Frame>(
        m,
        "Frame",
        "Camera pose and feature observations for single image frame"
    )
        .def(
            py::init<const Eigen::Isometry3d&, const CameraModel&>(),
            R"(Construct Frame given observed/measured camera pose and
               reference to camera)",
            py::arg("measured_camera_pose"),
            py::arg("camera")
        )
        .def(
            "addObservation",
            &Frame::addObservation,
            "Add feature observation to pose graph",
            py::arg("feature"),
            py::arg("observation")
        )
        .def(
            "measuredCameraPose",
            &Frame::measuredCameraPose,
            "Measured camera pose given when Frame was created"
        )
        .def(
            "cameraPose", &Frame::cameraPose, "Current estimate of camera pose"
        );

    py::classh<SLAM>(
        m,
        "SLAM",
        "SLAM from robot poses and image feature correspondences from "
        "robot-mounted camera"
    )
        .def(py::init<>())
        .def(
            "addFeature",
            &SLAM::addFeature,
            "Add new feature to pose graph",
            py::arg("position"),
            py::return_value_policy::reference_internal
        )
        .def(
            "addFrame",
            &SLAM::addFrame,
            "Add new Frame to pose graph",
            py::arg("frame")
        )
        .def(
            "setCameraFixed",
            &SLAM::setCameraFixed,
            "Set whether frame's camera pose is fixed or not",
            py::arg("frame"),
            py::arg("is_fixed")
        )
        .def(
            "solve",
            &SLAM::solve,
            R"(Run bundle-adjustment until convergence
 or maximum iterations are reached. Returns whether convergence was reached.)",
            py::arg("verbose_logging")
        );
}
