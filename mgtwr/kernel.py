import numpy as np
from scipy.spatial.distance import cdist


class GWRKernel:

    def __init__(
            self,
            coords: np.ndarray,
            bw: float = None,
            fixed: bool = True,
            function: str = 'triangular',
            eps: float = 1.0000001):

        self.coords = coords
        self.function = function
        self.bw = bw
        self.fixed = fixed
        self.function = function
        self.eps = eps
        self.bandwidth = None
        self.kernel = None

    def cal_distance(
            self,
            i: int):
        distance = cdist([self.coords[i]], self.coords).reshape(-1)
        return distance

    def cal_kernel(
            self,
            distance
    ):

        if self.fixed:
            self.bandwidth = float(self.bw)
        else:
            self.bandwidth = np.partition(
                distance,
                int(self.bw) - 1)[int(self.bw) - 1] * self.eps  # partial sort in O(n) Time

        self.kernel = self._kernel_funcs(distance / self.bandwidth)

        if self.function == "bisquare":  # Truncate for bisquare
            self.kernel[(distance >= self.bandwidth)] = 0
        return self.kernel

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * zs ** 2)
        elif self.function == 'bisquare':
            return (1 - zs ** 2) ** 2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)


class GTWRKernel(GWRKernel):

    def __init__(
            self,
            coords: np.ndarray,
            t: np.ndarray,
            bw: float = None,
            tau: float = None,
            fixed: bool = True,
            function: str = 'triangular',
            eps: float = 1.0000001):

        super(GTWRKernel, self).__init__(coords, bw, fixed=fixed, function=function, eps=eps)

        self.t = t
        self.tau = tau
        self.coords_new = None

    def cal_distance(
            self,
            i: int):

        if self.tau == 0:
            self.coords_new = self.coords
        else:
            self.coords_new = np.hstack([self.coords, (np.sqrt(self.tau) * self.t)])
        distance = cdist([self.coords_new[i]], self.coords_new).reshape(-1)
        return distance
