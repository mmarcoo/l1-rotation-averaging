# L1 Rotation Averaging

#### The code in this repository is not yet ready, it is still a work in progress.

I was looking for a simple yet robust Python implementation for rotations averaging and couldn't find one, so I tried to write one.

This repository contains an implementation of the rotation averaging algorithm described in [1] with some minor modifications to improve robustness borrowed by [2].

The core algorithm is based on [1] which computes the geodesic L1 median of rotations using the Weiszfeld algorithm on SO(3). The modifications from [2] include:

- A more robust initialization using the elementwise median
- An outlier rejection scheme at each Weiszfeld iteration

The goal of this package is to provide a simple and effective algorithm for averaging 3D rotations. While not as sophisticated as some other techniques, it offers a straightforward implementation of L1 rotation averaging that should be easy to use and integrate into other projects.

[1] Robust Rotation Averaging with Outlier Rejection, https://ieeexplore.ieee.org/abstract/document/5995745
[2] Robust L1 Norm Based Rotation Averaging, https://arxiv.org/pdf/2004.00732.pdf

## Installation

The package is not yet on PyPI, but you can install it from GitHub:

```bash
cd l1-rotation-averaging
pip install .
```

## Usage

The package:

- provides a toy command to try the algorithm on random data;
- can be installed and used as a simple library;
- contains a file `l1_rotation_averager/averager.py` that can be copy-pasted in your project, it depends only on `numpy`.

### Toy Command

To test the algorithm on randomly generated data:

```bash
l1-rotation-averaging --num-samples 100 --inliers-perc 0.5 --inliers-noise 15 --plot
```

This will generate 100 random rotations with 50 noisy (up to 15 degrees, normally distributed) and 50 completely random, take the L1 median, and display the results. See `l1-rotation-averaging --help` for all options.

### Simple Python Library 

The main class is `RotationAverager`, which can compute the L1 geodesic median of a list of rotation matrices:

```python
from l1_rotation_averaging import RotationAverager

R_list = [R1, R2, R3, ...]

averager = RotationAverager()
R_avg = averager.geodesic_L1_median(R_list)

```

The `RotationAverager` class allows configuring parameters like outlier rejection, number of iterations, convergence threshold etc. See the documentation for details.

### Copy-Paste
Opening the file `l1_rotation_averaging/averager.py` and copy-pasting the `RotationAverager` class in your project should be enough to use the algorithm. The class depends only on `numpy`, remember to import it!
