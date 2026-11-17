#ifndef KINEMATIC_BA_SLAM_FEATURE_HPP
#define KINEMATIC_BA_SLAM_FEATURE_HPP

#include <Eigen/Dense>

namespace kinematic_ba {
namespace slam {

// Tracked/unique feature point for SLAM
class Feature {
public:
    // Create new Feature given initial guess of 3D position
    Feature(const Eigen::Vector3d& position) : position(position) {}

    // Current 3D estimate of feature position in world/base base
    Eigen::Vector3d position;
};

}  // namespace slam
}  // namespace kinematic_ba

#endif
