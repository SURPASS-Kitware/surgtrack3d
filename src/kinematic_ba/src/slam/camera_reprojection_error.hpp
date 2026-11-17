#ifndef KINEMATIC_BA_SLAM_CAMERA_REPROJECTION_ERROR_HPP
#define KINEMATIC_BA_SLAM_CAMERA_REPROJECTION_ERROR_HPP

#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#include <ceres/cost_function.h>

#include <Eigen/Dense>
#include <memory>

#include "kinematic_ba/slam/camera_model.hpp"

namespace kinematic_ba {
namespace slam {

// Reprojection error from 2D observation/measurement for Ceres optimization
class CameraReprojectionError {
public:
    // observation is measured point in image coordinates,
    // camera is the intrinsics + local transform
    // Note: reference to camera is retained, so camera must outlive this
    CameraReprojectionError(
        const Eigen::Vector2d& observation,
        const CameraModel& camera,
        double stddev
    )
        : stddev(stddev), observation(observation), camera(&camera) {}

    // feature_ptr (size 3) is feature position in base frame
    // camera_p/q_ptr (size 3/4) is position/orientation of camera in base frame
    //
    // residuals_ptr (size 2) is
    // (i,j) error of observed vs. predicted point in image space
    template <typename T>
    bool operator()(
        const T* feature_ptr,
        const T* camera_p_ptr,
        const T* camera_q_ptr,
        T* residuals_ptr
    ) const {
        Eigen::Map<const Eigen::Vector3<T>> feature_p(feature_ptr);
        Eigen::Map<const Eigen::Vector3<T>> camera_p(camera_p_ptr);
        Eigen::Map<const Eigen::Quaternion<T>> camera_q(camera_q_ptr);

        // feature point in camera frame
        Eigen::Vector3<T> camera_point =
            camera_q.inverse() * (feature_p - camera_p);

        // object point projected in camera's image space
        Eigen::Vector2<T> camera_projection = camera->project(camera_point);

        Eigen::Map<Eigen::Vector2<T>> residuals(residuals_ptr);
        residuals = (camera_projection - observation.cast<T>()) / T(stddev);

        return true;
    }

    // see CameraReprojectionError() for argument details
    static std::unique_ptr<ceres::CostFunction> create(
        const Eigen::Vector2d& observation,
        const CameraModel& camera,
        double stddev
    ) {
        // 2 - size of residuals
        // 3 - size of feature position
        // 3 - size of camera position
        // 4 - size of camera orientation
        auto cost_functor = std::make_unique<CameraReprojectionError>(
            observation, camera, stddev
        );
        return std::make_unique<
            ceres::AutoDiffCostFunction<CameraReprojectionError, 2, 3, 3, 4>>(
            cost_functor.release()
        );
    }

private:
    const double stddev;  // isotropic standard deviation

    const Eigen::Vector2d observation;

    const CameraModel* camera;
};

}  // namespace slam
}  // namespace kinematic_ba

#endif
