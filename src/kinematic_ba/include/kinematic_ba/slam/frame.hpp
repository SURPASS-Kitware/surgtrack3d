#ifndef KINEMATIC_BA_SLAM_FRAME_HPP
#define KINEMATIC_BA_SLAM_FRAME_HPP

#include <Eigen/Geometry>

#include "camera_model.hpp"
#include "feature.hpp"

namespace kinematic_ba {
namespace slam {

// Camera pose and feature observations for single image frame
class Frame {
public:
    // Construct Frame given observed/measured camera pose and reference to camera
    Frame(
        const Eigen::Isometry3d& measured_camera_pose, const CameraModel& camera
    );

    // Add feature observation to pose graph
    void addObservation(Feature& feature, const Eigen::Vector2d& observation);

    // Measured camera pose given when Frame was created
    Eigen::Isometry3d measuredCameraPose() const;

    // Current estimate of camera pose
    Eigen::Isometry3d cameraPose() const;

    const CameraModel* camera;

    // Measured/observed camera pose, e.g. from measured robot pose +
    // eye-in-hand transform
    const Eigen::Vector3d camera_measured_p;
    const Eigen::Quaterniond camera_measured_q;

    // Modeled camera pose
    Eigen::Vector3d camera_p;
    Eigen::Quaterniond camera_q;

    std::vector<std::pair<Feature*, Eigen::Vector2d>> feature_observations;
};

}  // namespace slam
}  // namespace kinematic_ba

#endif
