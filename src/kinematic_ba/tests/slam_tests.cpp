#include <Eigen/Core>
#include <Eigen/Geometry>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>
#include <deque>
#include <numbers>

#include "kinematic_ba/slam/camera_model.hpp"
#include "kinematic_ba/slam/slam.hpp"

using namespace kinematic_ba::slam;
using namespace Catch::Matchers;

const int image_width = 1920;
const int image_height = 1080;

bool point_in_view(
    const Eigen::Vector3d& point,
    const Eigen::Isometry3d& camera_pose,
    const CameraModel& camera,
    double min_depth = 0.01
) {
    Eigen::Vector3d camera_point = camera_pose.inverse() * point;
    if (camera_point.z() < min_depth) { return false; }

    Eigen::Vector2d image_point = camera.project(camera_point);

    if (image_point.x() < 0 || image_point.x() >= image_width) { return false; }

    if (image_point.y() < 0 || image_point.y() >= image_height) {
        return false;
    }

    return true;
}

TEST_CASE("Perfect data SLAM", "[slam]") {
    SLAM slam;

    CameraModel camera(
        Eigen::Isometry3d::Identity(),
        Eigen::Projective2d(Eigen::Matrix3d(
            {{1200.0, 0.0, 960.0}, {0.0, 1200.0, 540.0}, {0.0, 0.0, 1.0}}
        ))
    );

    std::vector<Feature*> features;
    std::vector<Eigen::Vector3d> positions;

    for (int k = 0; k < 20; k++) {
        double theta = k / 20.0 * 2.0 * std::numbers::pi;
        positions.push_back(Eigen::Vector3d(
            0.03 * std::cos(theta),
            0.03 * std::sin(theta),
            0.06 * (1.0 + (k - 10.0) * (k - 10.0) / 400.0)
        ));
        Feature* feature =
            slam.addFeature(0.06 * Eigen::Vector3d(std::cos(theta), 0.0, 1.0));
        features.push_back(feature);
    }

    // Add three frames
    std::deque<Frame> frames;
    std::vector<Eigen::Isometry3d> camera_poses;
    for (int k = 0; k < 9; k++) {
        double x = (k - 4.0) / 4.0;
        Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
        pose.translation().x() = 0.05 * x;
        pose.translation().y() = 0.05 * x * x - 0.025;
        camera_poses.push_back(pose);

        Frame& frame = frames.emplace_back(pose, camera);

        for (size_t idx = 0; idx < features.size(); idx++) {
            if (point_in_view(positions[idx], pose, camera)) {
                Eigen::Vector2d projection =
                    camera.project(pose.inverse() * positions[idx]);
                frame.addObservation(*features[idx], projection);
            }
        }

        slam.addFrame(frame);
    }

    bool success = slam.solve(false);

    REQUIRE(success);

    for (size_t k = 0; k < camera_poses.size(); k++) {
        const Eigen::Isometry3d pose = camera_poses[k];

        double translation_error =
            (pose.translation() - frames[k].cameraPose().translation()).norm();
        double rotation_error =
            ((pose.rotation() * frames[k].cameraPose().rotation().inverse())
                 .matrix() -
             Eigen::Matrix3d::Identity())
                .norm();
        REQUIRE_THAT(translation_error, WithinAbs(0.0, 1.0e-6));
        REQUIRE_THAT(rotation_error, WithinAbs(0.0, 1.0e-6));
    }

    // Verify feature positions are correct
    for (int k = 0; k < 20; k++) {
        CAPTURE(k, positions[k], features[k]->position);
        REQUIRE_THAT(
            (positions[k] - features[k]->position).norm(),
            WithinAbs(0.0, 1.0e-6)
        );
    }
}
