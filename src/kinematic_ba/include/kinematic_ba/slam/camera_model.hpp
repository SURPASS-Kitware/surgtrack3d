#ifndef KINEMATIC_BA_SLAM_CAMERA_MODEL_HPP
#define KINEMATIC_BA_SLAM_CAMERA_MODEL_HPP

#include <Eigen/Geometry>

namespace kinematic_ba {
namespace slam {

// Rectified camera projection model
class CameraModel {
public:
    // Construct from camera pose (optical frame wrt camera frame) and
    // (rectified) camera intrinsic matrix. Optical frame may be different from
    // camera frame for e.g. stereo cameras, where both camera heads share a
    // single camera reference frame.
    CameraModel(
        const Eigen::Isometry3d& pose, const Eigen::Projective2d& intrinsic
    ) {
        projection.noalias() = intrinsic.matrix() * pose.inverse().affine();
    }

    // Construct directly from camera projection matrix
    CameraModel(const Eigen::Matrix<double, 3, 4>& projection)
        : projection(projection) {}

    // Parameterized by numeric type T for autodiff.
    template <typename T>
    Eigen::Vector<T, 2> project(const Eigen::Vector<T, 3>& point) const {
        Eigen::Vector<T, 3> projected = projection * point.homogeneous();

        return projected.template head<2>() / projected.z();
    }

    const Eigen::Matrix<double, 3, 4>& projectionMatrix() const {
        return projection;
    }

private:
    Eigen::Matrix<double, 3, 4> projection;
};

}  // namespace slam
}  // namespace kinematic_ba

#endif