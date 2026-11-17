#ifndef KINEMATIC_BA_SLAM_SLAM_HPP
#define KINEMATIC_BA_SLAM_SLAM_HPP

#include <ceres/problem.h>

#include <Eigen/Geometry>
#include <deque>

#include "feature.hpp"
#include "frame.hpp"

namespace kinematic_ba {
namespace slam {

class SLAM {
public:
    // Add new feature to pose graph
    Feature* addFeature(const Eigen::Vector3d& position);

    // Add new Frame to pose graph
    void addFrame(Frame& frame);

    void setCameraFixed(Frame& frame, bool is_fixed);

    // Run bundle-adjustment until convergence or maximum iterations are
    // reached. Returns whether convergence was reached.
    bool solve(bool verbose_logging);

private:
    // std::deque used to prevent re-allocated, so Ceres can uniquely identify
    // features by pointer/address
    std::deque<Feature> features;
    std::deque<Frame*> frames;

    ceres::Problem problem;

    void addPoseParameters(
        Eigen::Vector3d& position, Eigen::Quaterniond& orientaton, bool constant
    );
};

}  // namespace slam
}  // namespace kinematic_ba

#endif
