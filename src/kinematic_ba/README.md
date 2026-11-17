# kinematic_ba

Bundle adjustment for surgical stereo video, built on the [Ceres](http://ceres-solver.org/) optimizer. Camera poses from the robot's kinematics act as priors, and 2D feature tracks are jointly optimized with them to produce refined camera poses and a sparse 3D point cloud.

The core is a C++ library (`include/`, `src/`), exposed to Python via pybind11 (`bindings/`) as the `kinematic_ba` package:

```python
from kinematic_ba import slam

solver = slam.SLAM()
camera = slam.CameraModel(...)
frame = slam.Frame(pose, camera)
```

See `fast_tissue_rec/run_slam.py` for a complete usage example.

- [Layout](#layout)
- [Dependencies](#dependencies)
- [Building the Python module](#building-the-python-module)
- [Editable install](#editable-install)
- [Building and running the C++ tests](#building-and-running-the-c-tests)

## Layout

```
kinematic_ba/
├── CMakeLists.txt  CMakePresets.json  vcpkg.json   C++ build (vcpkg manifest)
├── pyproject.toml                                  Python packaging (scikit-build-core)
├── include/kinematic_ba/slam/                      public headers
├── src/slam/                                       implementation + Ceres cost functions
├── bindings/                                       pybind11 module (_core)
├── kinematic_ba/__init__.py                        Python package
└── tests/                                          Catch2 unit tests
```

## Dependencies

This package is built and tested on Ubuntu 24.04, with the system version of Python (3.12). Newer releases may work. Unlike the rest of the pipeline, it does **not** require CUDA.

You will need to have [vcpkg](https://vcpkg.io) installed, and the `VCPKG_ROOT` environment variable must be defined. vcpkg builds Ceres (with SuiteSparse), Eigen, pybind11 and Catch2 from `vcpkg.json` automatically during the first build. You may also need to install a compatible Fortran compiler for building `lapack`, see https://github.com/microsoft/vcpkg/issues/46242#issuecomment-3203053500.

On Linux, you may need to install `autoconf`, `libtoolize`, and `automake` if you don't have them already --- on Debian/Ubuntu this can be done via `sudo apt install autoconf autoconf-archive automake libtool`.

## Building the Python module

To build the Python module, you will also need to install Python's `build` tool. This can be done via `pip install build`, or on Ubuntu via `sudo apt install python3-build` to install system-wide.

From this directory (`src/kinematic_ba`), run
```sh
python -m build --wheel
```
This will generate the Python bindings for the C++ library in the `dist/` folder, which you can install using `pip install --force-reinstall ./dist/kinematic_ba-<...>.whl`.

Alternatively, install directly with
```sh
pip install ./src/kinematic_ba        # from the repository root
```

## Editable install

Note: it is *not* recommended to use an editable install for non-development purposes.

Note: it is recommended to set `PYTHONPYCACHEPREFIX` in your `.bashrc`, or else editable install will litter your source tree with `__pycache__` folders.

Alternatively, you can do an "editable install" so that changes to source files are immediately reflected without you having to rebuild the module. Inside this directory (and with the venv activated) run:
```sh
pip install -e . --no-build-isolation
```
to do an editable build.

## Building and running the C++ tests

The Catch2 tests are built when CMake is invoked directly (not through pip):
```sh
cmake --preset vcpkg -B build -S .
cmake --build build
./build/tests/tests
```
