import gc
import importlib
import logging
import pathlib
import typing
from collections.abc import Sequence

import cupy
import numpy as np
import numpy.typing as npt
import tensorrt as trt
import yaml
from cuda.bindings import runtime as cudart
from tqdm import tqdm

from fast_tissue_rec import stereo_utils


class TRTStereoEngine:
    """TensorRT runner for Fast-FoundationStereo engine"""

    def __init__(
        self,
        engine_path: pathlib.Path,
        image_size: tuple[int, int],
        camera_calibration: dict,
    ):
        self.baseline = stereo_utils.get_stereo_baseline(camera_calibration)
        self.fx = stereo_utils.left_camera_matrix(camera_calibration)[0, 0]

        print(f"Baseline: {self.baseline}, fx: {self.fx}")

        self.input_shape = (*image_size, 3)

        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        err, self.stream = cudart.cudaStreamCreate()
        if err != cudart.cudaError_t.cudaSuccess:
            print("Error when creating dedicated CUDA stream: ", err)

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize TRT engine from {engine_path}. "
                f"This usually means the engine was built with a different "
                f"TensorRT version (yours: {trt.__version__}). "
                f"Rebuild with:  trtexec --onnx=<your .onnx> "
                f"--saveEngine={engine_path} --fp16"
            )
        self.context = self.engine.create_execution_context()

    def release(self):
        del self.context
        del self.engine
        cudart.cudaStreamDestroy(self.stream)
        del self.stream

        gc.collect()

        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()

    @staticmethod
    def load(camera_calibration: dict) -> typing.Self:
        model_path = importlib.resources.files("fast_tissue_rec.resources").joinpath(
            "finetuned_foundation_stereo_32_iters.engine"
        )
        config_path = importlib.resources.files("fast_tissue_rec.resources").joinpath(
            "fast_foundationstereo.yaml"
        )
        # model_path = importlib.resources.files(
        #     "fast_tissue_rec.resources"
        # ).joinpath("fastfs_v7_ep6_1920x1056_fp16.engine")
        # config_path = importlib.resources.files(
        #     "fast_tissue_rec.resources"
        # ).joinpath("fastfs_v7_ep6_1920x1056_fp16.yaml")

        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        target_h, target_w = cfg["image_size"]
        logging.info(f"Model target resolution: {target_h} x {target_w}")

        # Load TRT engine
        logging.info(f"Loading model: {model_path}")
        model = TRTStereoEngine(model_path, (target_h, target_w), camera_calibration)
        print("TensorRT stereo engine loaded")

        return model

    def _trt_to_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT: cupy.float32,
            trt.DataType.HALF: cupy.float16,
            trt.DataType.INT32: cupy.int32,
            trt.DataType.INT8: cupy.int8,
        }
        if dt not in mapping:
            raise RuntimeError(f"Unsupported TRT dtype: {dt}")
        return mapping[dt]

    def run_inference(
        self, left_image: cupy.ndarray, right_image: cupy.ndarray
    ) -> npt.NDArray:
        """Run inference."""
        trt = self.trt

        inputs = {"left": left_image, "right": right_image}

        for name, tensor in inputs.items():
            expected = self._trt_to_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                print(
                    f"!! For input {name}, expected type {expected}, but received {tensor.dtype}"
                )
                inputs[name] = tensor.astype(expected)
            if not tensor.flags.c_contiguous:
                print(f"!! {name} is not a contiguous tensor")
                inputs[name] = cupy.ascontiguousarray(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))

        out_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
            == trt.TensorIOMode.OUTPUT
        ]

        outputs = {}
        for name in out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = cupy.empty(shape, dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data.ptr))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data.ptr))

        assert self.context.execute_async_v3(self.stream)

        cupy.cuda.ExternalStream(int(self.stream)).synchronize()

        return outputs["disp"]

    def stereo_to_depth(self, left_image, right_image):
        left_raw_gpu = cupy.asarray(left_image, dtype=cupy.float32)
        left_tensor = left_raw_gpu.reshape((1,) + left_raw_gpu.shape)
        left_tensor = cupy.ascontiguousarray(left_tensor.transpose(0, 3, 1, 2))

        right_raw_gpu = cupy.asarray(right_image, dtype=cupy.float32)
        right_tensor = right_raw_gpu.reshape((1,) + right_raw_gpu.shape)
        right_tensor = cupy.ascontiguousarray(right_tensor.transpose(0, 3, 1, 2))

        disparity = self.run_inference(left_tensor, right_tensor)
        H, W = left_image.shape[0], left_image.shape[1]
        disparity = disparity.reshape(H, W).clip(0, None).get()

        depth = stereo_utils.depth_from_disparity(
            disparity=disparity, fx=self.fx, baseline=self.baseline
        )

        return depth


def stereo_images_to_depth_maps(
    image_pairs: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    camera_calibration: dict,
) -> Sequence[npt.NDArray[np.float32]]:
    """Given sequence of pairs of rectified RGB stereo images and stereo camera
    calibration, produces corresponding sequence of depth maps. Depth maps are Numpy
    MaskedArray's, and may contain masked-out invalid regions.
    """
    depth_estimator = TRTStereoEngine.load(camera_calibration)

    depth_maps = []

    for frame in tqdm(image_pairs):
        for img in frame:
            if img.shape != depth_estimator.input_shape:
                print(
                    f"Stereo depth estimation input shape mismatch: expected"
                    f"{depth_estimator.input_shape}, but received {img.shape}"
                )
                return None

        predicted_depth = depth_estimator.stereo_to_depth(frame[0], frame[1])
        depth_maps.append(predicted_depth)

    depth_estimator.release()

    return depth_maps
