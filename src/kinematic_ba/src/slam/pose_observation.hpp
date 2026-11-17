#ifndef KINEMATIC_BA_SLAM_POSE_OBSERVATION_HPP
#define KINEMATIC_BA_SLAM_POSE_OBSERVATION_HPP

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>
#include <ceres/rotation.h>

#include <Eigen/Geometry>
#include <memory>

namespace kinematic_ba {
namespace slam {

// Adapted from version of Ceres that is more recent than is currently available
// from vcpkg, original copyright Google 2025 - see include/ceres/rotation.h in
// version 0ba987aca or later.
template <typename T>
inline void QuaternionToAngleAxis(
    const Eigen::Quaternion<T>& quaternion, T* angle_axis
) {
    using std::fpclassify;
    using std::hypot;
    const T& q1 = quaternion.x();
    const T& q2 = quaternion.y();
    const T& q3 = quaternion.z();

    T k;
    const T sin_theta = hypot(q1, q2, q3);

    // For quaternions representing non-zero rotation, the conversion
    // is numerically stable.
    if (fpclassify(sin_theta) != FP_ZERO) {
        const T& cos_theta = quaternion.w();

        // If cos_theta is negative, theta is greater than pi/2, which
        // means that angle for the angle_axis vector which is 2 * theta
        // would be greater than pi.
        //
        // While this will result in the correct rotation, it does not
        // result in a normalized angle-axis vector.
        //
        // In that case we observe that 2 * theta ~ 2 * theta - 2 * pi,
        // which is equivalent saying
        //
        //   theta - pi = atan(sin(theta - pi), cos(theta - pi))
        //              = atan(-sin(theta), -cos(theta))
        //
        const T sign = copysign(T(1), cos_theta);
        const T two_theta = T(2.0) * atan2(sign * sin_theta, sign * cos_theta);
        k = two_theta / sin_theta;
    } else {
        // For zero rotation, sqrt() will produce NaN in the derivative since
        // the argument is zero.  By approximating with a Taylor series,
        // and truncating at one term, the value and first derivatives will be
        // computed correctly when Jets are used.
        k = T(2.0);
    }

    angle_axis[0] = q1 * k;
    angle_axis[1] = q2 * k;
    angle_axis[2] = q3 * k;
}

// Error/distance between modelled and observed poses for Ceres optimization
class PoseObservation {
public:
    // observation is measured/observed pose
    // position_stddev is isotropic standard deviation in R^3
    // orientation_stddev is isotropic standard deviation in so(3)
    PoseObservation(
        const Eigen::Isometry3d& observation,
        double position_stddev,
        double orientation_stddev
    )
        : observation(observation),
          orientation(observation.rotation()),
          position_stddev(position_stddev),
          orientation_stddev(orientation_stddev) {}

    // p/q_ptr (size 3/4) is modeled position/orientation
    // residuals_ptr (size 6) is position error R^3 and orientation error so(3)
    template <typename T>
    bool operator()(const T* p_ptr, const T* q_ptr, T* residuals_ptr) const {
        Eigen::Map<const Eigen::Vector3<T>> p(p_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> q(q_ptr);

        Eigen::Map<Eigen::Vector<T, 6>> residual(residuals_ptr);
        residual.template head<3>() =
            (observation.cast<T>().translation() - p) / T(position_stddev);

        Eigen::Quaternion<T> rotation_error =
            orientation.conjugate().cast<T>() * q;
        slam::QuaternionToAngleAxis(
            rotation_error, residual.template tail<3>().data()
        );
        residual.template tail<3>() /= T(orientation_stddev);

        return true;
    }

    // see PoseObservation() for argument details
    static std::unique_ptr<ceres::CostFunction> create(
        const Eigen::Isometry3d& observation,
        double position_stddev,
        double orientation_stddev
    ) {
        // 6 - size of residuals
        // 3 - size of position
        // 4 - size of orientation
        auto cost_functor = std::make_unique<PoseObservation>(
            observation, position_stddev, orientation_stddev
        );
        return std::make_unique<
            ceres::AutoDiffCostFunction<PoseObservation, 6, 3, 4>>(
            cost_functor.release()
        );
    }

private:
    // observed pose, with error
    const Eigen::Isometry3d observation;
    const Eigen::Quaterniond orientation;

    double position_stddev;
    double orientation_stddev;
};

}  // namespace slam
}  // namespace kinematic_ba

#endif
