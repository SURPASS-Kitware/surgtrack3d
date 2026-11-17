import cv2 as cv
import numpy as np
import numpy.typing as npt
from typing import Self


def left_camera_matrix(camera_calibration: dict) -> npt.NDArray[np.float32]:
    """Get 3x3 camera matrix for rectified left camera from camera calibration"""
    projection = np.array(camera_calibration["left_camera"]["projection"])[:3, :3]
    return projection


def get_stereo_baseline(
    camera_calibration: dict
) -> float:
    """Get stereo baseline from camera calibration"""
    projection_matrix = np.array(camera_calibration["right_camera"]["projection"])

    baseline_meters = -projection_matrix[0, 3] / projection_matrix[0, 0]
    print("Baseline (mm): ", 1000.0 * baseline_meters)
    return baseline_meters


def depth_from_disparity(
    disparity: npt.NDArray, fx: float, baseline: float
) -> npt.NDArray:
    yy, xx = np.meshgrid(
        np.arange(disparity.shape[0]), np.arange(disparity.shape[1]), indexing="ij"
    )
    us_right = xx - disparity
    invalid = (us_right < 0) | (disparity == 0.0)

    depth = fx * baseline / disparity
    depth[invalid] = 0

    return depth


class ImageDownsampler:
    @staticmethod
    def square(camera_calibration: dict, in_size: tuple[float, float], out_size: float) -> Self:
        in_width, in_height = in_size
        square_size = min(in_width, in_height)
        start_x = (in_width - square_size) // 2
        start_y = (in_height - square_size) // 2

        crop_x = (start_x, square_size + start_x)
        crop_y = (start_y, square_size + start_y)

        return ImageDownsampler(camera_calibration, crop_x, crop_y, out_size, out_size)

    def __init__(self, camera_calibration: dict, crop_x: tuple[float, float], crop_y: tuple[float, float], out_width: float, out_height: float):
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.out_width = out_width
        self.out_height = out_height

        self.calibration = {
            "left_camera": self._downsample_calibration(camera_calibration["left_camera"]),
            "right_camera": self._downsample_calibration(camera_calibration["right_camera"]),
        }

    def _crop_calibration(self, camera_matrix: npt.NDArray) -> npt.NDArray:
        cropped = camera_matrix.copy()
        cropped[0][2] -= self.crop_x[0]
        cropped[1][2] -= self.crop_y[0]
        return cropped

    def _scale_calibration(self, scale_x: float, scale_y: float, camera_matrix: npt.NDArray) -> npt.NDArray:
        scaled = camera_matrix.copy()
        scaled[0, :] *= scale_x
        scaled[1, :] *= scale_y
        return scaled        

    def _downsample_calibration(self, calibration: dict) -> dict:
        scale_x = self.out_width / (self.crop_x[1] - self.crop_x[0])
        scale_y = self.out_height / (self.crop_y[1] - self.crop_y[0])

        def downsample(camera_matrix):
            camera_matrix = np.array(camera_matrix)
            cropped = self._crop_calibration(camera_matrix)
            scaled = self._scale_calibration(scale_x, scale_y, cropped)
            return scaled.tolist()

        output_calibration = {
            "image_height": self.out_height,
            "image_width": self.out_width,
            "distortion": calibration["distortion"],
            "rectification": calibration["rectification"],
            "intrinsic": downsample(calibration["intrinsic"]),
            "projection": downsample(calibration["projection"]),
        }

        return output_calibration

    def __call__(self, image):
        cropped = image[self.crop_y[0]:self.crop_y[1], self.crop_x[0]:self.crop_x[1]]
        scaled = cv.resize(cropped, (self.out_width, self.out_height), interpolation=cv.INTER_AREA)
        return scaled
