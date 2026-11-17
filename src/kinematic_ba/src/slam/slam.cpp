#include "kinematic_ba/slam/slam.hpp"

#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/manifold.h>
#include <ceres/types.h>

#include <numbers>

#include "camera_reprojection_error.hpp"
#include "pose_observation.hpp"

namespace kinematic_ba {
namespace slam {

Feature* SLAM::addFeature(const Eigen::Vector3d& position) {
    features.emplace_back(position);
    return &features.back();
};

void SLAM::addFrame(Frame& frame) {
    addPoseParameters(frame.camera_p, frame.camera_q, false);

    std::unique_ptr<ceres::CostFunction> pose_cost_function =
        PoseObservation::create(
            frame.measuredCameraPose(), 0.005, 5.0 * std::numbers::pi / 180.0
        );
    problem.AddResidualBlock(
        pose_cost_function.release(),
        new ceres::HuberLoss(1.0),
        frame.camera_p.data(),
        frame.camera_q.coeffs().data()
    );

    for (auto& feature : frame.feature_observations) {
        std::unique_ptr<ceres::CostFunction> feature_reprojection_cost =
            CameraReprojectionError::create(feature.second, *frame.camera, 6.0);
        problem.AddResidualBlock(
            feature_reprojection_cost.release(),
            new ceres::HuberLoss(1.0),
            feature.first->position.data(),
            frame.camera_p.data(),
            frame.camera_q.coeffs().data()
        );
    }

    frames.push_back(&frame);
}

void SLAM::setCameraFixed(Frame& frame, bool is_fixed) {
    if (is_fixed) {
        problem.SetParameterBlockConstant(frame.camera_p.data());
        problem.SetParameterBlockConstant(frame.camera_q.coeffs().data());
    } else {
        problem.SetParameterBlockVariable(frame.camera_p.data());
        problem.SetParameterBlockVariable(frame.camera_q.coeffs().data());
    }
}

void SLAM::addPoseParameters(
    Eigen::Vector3d& position, Eigen::Quaterniond& orientaton, bool constant
) {
    problem.AddParameterBlock(position.data(), 3);
    // the ceres::Problem instance will take ownership of the manifold
    ceres::Manifold* quaternion_manifold = new ceres::EigenQuaternionManifold;
    problem.AddParameterBlock(orientaton.coeffs().data(), 4);
    problem.SetManifold(orientaton.coeffs().data(), quaternion_manifold);

    if (constant) {
        problem.SetParameterBlockConstant(position.data());
        problem.SetParameterBlockConstant(orientaton.coeffs().data());
    }
}

bool SLAM::solve(bool verbose_logging) {
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
    options.max_num_iterations = 500;

    options.minimizer_progress_to_stdout = verbose_logging;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose_logging) { std::cout << summary.FullReport() << std::endl; }

    return summary.termination_type == ceres::TerminationType::CONVERGENCE;
}

}  // namespace slam
}  // namespace kinematic_ba
