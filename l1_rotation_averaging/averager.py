import numpy as np


class RotationAverager:
    def __init__(
        self,
        reject_outliers: bool = True,
        iterations: int = 10,
        thr_convergence: float = 0.001,
        outliers_min_angle: float = 30,
    ):
        """This is a Python implementation of the rotation averaging algorithm
        described in [1] https://ieeexplore.ieee.org/abstract/document/5995745 with some
        minor modifications to improve robustness borrowed by [2] https://arxiv.org/pdf/2004.00732.pdf

        Args:
            reject_outliers (bool, optional): Decide whether using the original
                algorithm [1] or the modified one [2]. Defaults to True.
            iterations (int, optional): Number of maximum iterations. Defaults to 10.
            thr_convergence (float, optional): Threshold to exit the optimization loop
                before reaching maximum iterations. Defaults to 0.001.
            outliers_min_angle (float, optional): The minimum angle in degrees for which
                a value is considered an outlier. When you have few good observations
                (or very low noise) you risk to reject almost all inliers, in this way
                you keep all the inliers up to this angle. Defaults to 30.
        """
        self._outliers = reject_outliers
        self._iters = iterations
        self._thr_convergence = thr_convergence
        self._outliers_min_angle = np.radians(outliers_min_angle)

    def _get_thr(
        self,
        norms: np.ndarray,
    ) -> float:
        # Here I get the 25th percentile of the norms
        # aka the angle that separates the 25% smallest norms from the 75% largest norms
        best_norms = np.quantile(norms, 0.25).astype(float)

        # If this angle is smaller than the minimum angle, then I use the minimum angle
        # this is to avoid the case where the 25th percentile is very small and I end up
        # rejecting a lot of inliers
        thr = max(best_norms, self._outliers_min_angle)

        return thr

    def _log(self, R: np.ndarray) -> np.ndarray:
        # The cosine of the rotation angle is related to the trace of C
        cos_angle = 0.5 * np.trace(R) - 0.5
        # Clip cos(angle) to its proper domain to avoid NaNs from rounding errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        # If angle is close to zero, use first-order Taylor expansion
        if np.isclose(angle, 0.0):
            return self._skew_to_vec(R - np.identity(3))

        # Otherwise take the matrix logarithm and return the rotation vector
        return self._skew_to_vec((0.5 * angle / np.sin(angle)) * (R - R.T))

    def _vec_to_skew(self, phi: np.ndarray) -> np.ndarray:
        phi = np.atleast_2d(phi)
        if phi.shape[1] != 3:
            raise ValueError("phi must have shape ({},) or (N,{})".format(3, 3))

        Phi = np.zeros([phi.shape[0], 3, 3])
        Phi[:, 0, 1] = -phi[:, 2]
        Phi[:, 1, 0] = phi[:, 2]
        Phi[:, 0, 2] = phi[:, 1]
        Phi[:, 2, 0] = -phi[:, 1]
        Phi[:, 1, 2] = -phi[:, 0]
        Phi[:, 2, 1] = phi[:, 0]

        return np.squeeze(Phi)

    def _skew_to_vec(self, Phi: np.ndarray) -> np.ndarray:
        if Phi.ndim < 3:
            Phi = np.expand_dims(Phi, axis=0)

        if Phi.shape[1:3] != (3, 3):
            raise ValueError(
                "Phi must have shape ({},{}) or (N,{},{})".format(3, 3, 3, 3)
            )

        phi = np.empty([Phi.shape[0], 3])
        phi[:, 0] = Phi[:, 2, 1]
        phi[:, 1] = Phi[:, 0, 2]
        phi[:, 2] = Phi[:, 1, 0]
        return np.squeeze(phi)

    def _exp(self, phi):
        if len(phi) != 3:
            raise ValueError("phi must have length 3")

        angle = np.linalg.norm(phi)

        # Near phi==0, use first order Taylor expansion
        if np.isclose(angle, 0.0):
            return np.identity(3) + self._vec_to_skew(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)

        return (
            c * np.identity(3)
            + (1 - c) * np.outer(axis, axis)
            + s * self._vec_to_skew(axis)
        )

    def geodesic_L1_median(self, R: list[np.ndarray]) -> np.ndarray:
        # Initialization: median of the input rotations + SVD
        r = np.vstack([R[i].flatten() for i in range(len(R))]).astype(np.float64)
        s_0: np.ndarray = np.median(r, axis=0)

        U, _, V = np.linalg.svd(s_0.reshape(3, 3))
        S_0 = U @ V

        # Optimize
        S_t = S_0
        for _ in range(self._iters):
            vs = np.array([self._log(R[i] @ S_t.T) for i in range(len(R))])
            norms = np.linalg.norm(vs, axis=1, keepdims=True)

            # Compute inlier threshold (only if outliers rejection is enabled)
            thr = self._get_thr(norms) if self._outliers else np.inf

            # Compute Weiszfeld step
            wi = np.where(norms <= thr, 1, 0)
            delta = np.sum(wi * vs / norms, axis=0) / np.sum(wi / norms, axis=0)

            S_t = self._exp(delta) @ S_t

            if np.linalg.norm(delta) < self._thr_convergence:
                break

        S_opt = S_t
        return S_opt
