"""Top-level package for l1-rotation-averaging."""

__author__ = "Marco Toschi"
__email__ = "marco.toschi@eyecan.ai"
__version__ = "0.0.1"

import random
from typing import Optional

import numpy as np
import typer as t
import visu3d as v3d
from rich.console import Console
from rich.table import Table
from scipy.spatial.transform import Rotation

from .averager import RotationAverager

app = t.Typer()


@app.command("random")
def average_rotation_test(
    num_samples: int = t.Option(..., help="Number of samples"),
    inliers_perc: float = t.Option(0.4, help="Percentage of inliers"),
    inliers_noise: float = t.Option(10, help="Inliers noise level (deg)"),
    outliers_rejection: bool = t.Option(True, help="Outliers rejection"),
    iterations: int = t.Option(10, help="Number of iterations"),
    threshold: float = t.Option(0.001, help="Convergence threshold"),
    seed: Optional[int] = t.Option(None, help="Random seed"),
    plot: bool = t.Option(True, help="Plot"),
):
    """
    Average a set of rotations using geodesic L1 mean.
    """

    np.random.seed(seed)

    n_inliers = int(num_samples * inliers_perc)
    n_outliers = num_samples - n_inliers

    R_true = Rotation.random(1).as_matrix().squeeze(0)

    # 1. Create input rotations:
    n_samples = n_inliers + n_outliers

    R_inliers = []
    R_outliers = []

    for i in range(n_samples):
        if i < n_inliers:
            # Inliers: perturb by chosen deg.
            rand_axis = np.random.rand(3) - 0.5
            rand_axis_unit = rand_axis / np.linalg.norm(rand_axis)
            angle_perturb = np.random.normal(0, inliers_noise / 180 * np.pi)
            R_perturb = Rotation.from_rotvec(angle_perturb * rand_axis_unit).as_matrix()
            R_inliers.append(R_perturb @ R_true)
        else:
            # Outliers: completely random.
            R_outliers.append(Rotation.random(1).as_matrix().squeeze(0))

    R_samples = R_inliers + R_outliers

    # Shuffle the order of the samples
    random.shuffle(R_samples)

    # 2. Averaging
    averager = RotationAverager(outliers_rejection, iterations, threshold)
    R_geodesic = averager.geodesic_L1_median(R_samples)  # list of 3x3 matrices

    # 3. Evaluate rotation error (deg)
    err_geodesic = np.arccos((np.trace((R_true @ R_geodesic.T)) - 1) / 2) * 180 / np.pi

    # 4. Print results
    params = {
        "num_samples": num_samples,
        "inliers_perc": inliers_perc,
        "inliers_noise": inliers_noise,
        "outliers_rejection": outliers_rejection,
        "iterations": iterations,
        "threshold": threshold,
        "seed": seed,
    }

    # Create console
    console = Console()
    table = Table(title="Rotation Averaging Results")
    table.add_column("Parameters", justify="right", style="cyan")
    table.add_column("Ground Truth", justify="right", style="green")
    table.add_column("Predicted", justify="right", style="yellow")
    table.add_column("Error (degrees)", justify="right", style="red")
    table.add_row(str(params), str(R_true), str(R_geodesic), str(err_geodesic))

    console.print(table)

    # 5. Visualize
    if plot:
        spec = v3d.PinholeCamera.from_focal(resolution=(256, 144), focal_in_px=1024)
        to_plot = {
            "R_noisy_observations": R_inliers,
            "R_wrong_observations": R_outliers,
            "R_estimation": np.expand_dims(R_geodesic, axis=0),
            "R_ground_truth": np.expand_dims(R_true, axis=0),
        }
        cams_list = []
        for key, poses in to_plot.items():
            # # Create a Camera looking at the center
            trans = v3d.Transform(R=poses, t=np.array([0, 0, 0]))  # type: ignore
            cams = v3d.Camera(spec=spec, world_from_cam=trans)
            cams = cams.replace_fig_config(name=key, scale=0.1)
            cams_list.append(cams)

        fig = v3d.make_fig(cams_list)
        fig.show()
