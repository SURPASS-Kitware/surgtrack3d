#include "kinematic_ba/slam/frame.hpp"

namespace kinematic_ba {
namespace slam {

Frame::Frame(
    const Eigen::Isometry3d& measured_camera_pose, const CameraModel& camera
)
    : camera(&camera),
      camera_measured_p(measured_camera_pose.translation()),
      camera_measured_q(measured_camera_pose.rotation()),
      camera_p(camera_measured_p),
      camera_q(camera_measured_q) {}

void Frame::addObservation(
    Feature& feature, const Eigen::Vector2d& observation
) {
    feature_observations.emplace_back(&feature, observation);
}

Eigen::Isometry3d Frame::measuredCameraPose() const {
    Eigen::Isometry3d pose;
    pose.translation() = camera_measured_p;
    pose.linear() = camera_measured_q.toRotationMatrix();
    return pose;
}

Eigen::Isometry3d Frame::cameraPose() const {
    Eigen::Isometry3d pose;
    pose.translation() = camera_p;
    pose.linear() = camera_q.toRotationMatrix();
    return pose;
}

}  // namespace slam
}  // namespace kinematic_ba
