from typing import Union
import numpy as np
import pandas as pd
import multiprocessing as mp
from .kernel import GWRKernel, GTWRKernel
from .function import _compute_betas_gwr, surface_to_plane
from .obj import CalAicObj, CalMultiObj, BaseModel, GWRResults, GTWRResults, MGWRResults, MGTWRResults


class GWR(BaseModel):
    """
    Geographically Weighted Regression
    """
    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame, pd.Series],
            bw: float,
            kernel: str = 'bisquare',
            fixed: bool = True,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False,
    ):
        """
        Parameters
        ----------
        coords        : array-like
                        n*2, spatial coordinates of the observations, if it's latitude and longitude,
                        the first column should be longitude

        X             : array-like
                        n*k, independent variable, excluding the constant

        y             : array-like
                        n*1, dependent variable

        bw            : scalar
                        bandwidth value consisting of either a distance or N
                        nearest neighbors; user specified or obtained using
                        sel

        kernel        : string
                        type of kernel function used to weight observations;
                        available options:
                        'gaussian'
                        'bisquare'
                        'exponential'

        fixed         : bool
                        True for distance based kernel function and  False for
                        adaptive (nearest neighbor) kernel function (default)

        constant      : bool
                        True to include intercept (default) in model and False to exclude
                        intercept.

        thread        : int
                        The number of processes in parallel computation. If you have a large amount of data,
                        you can use it

        convert       : bool
                        Whether to convert latitude and longitude to plane coordinates.
        Examples
        --------
        import numpy as np
        from mgtwr.model import GWR
        np.random.seed(10)
        u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
        v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
        t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
        x1 = np.random.uniform(0, 1, (1728, 1))
        x2 = np.random.uniform(0, 1, (1728, 1))
        epsilon = np.random.randn(1728, 1)
        beta0 = 5
        beta1 = 3 + (u + v + t)/6
        beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
        y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
        coords = np.hstack([u, v])
        X = np.hstack([x1, x2])
        gwr = GWR(coords, X, y, 0.8, kernel='gaussian', fixed=True).fit()
        print(gwr.R2)
        0.7128737240047688
        """
        super(GWR, self).__init__(X, y, kernel, fixed, constant)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.bw = bw
        self.thread = thread

    def _build_wi(self, i, bw):
        """
        calculate Weight matrix
        """
        try:
            gwr_kernel = GWRKernel(self.coords, bw, fixed=self.fixed, function=self.kernel)
            distance = gwr_kernel.cal_distance(i)
            wi = gwr_kernel.cal_kernel(distance)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def cal_aic(self):
        """
        use for calculating AICc, BIC, CV and so on.
        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._search_local_fit, range(self.n))))
        else:
            result = list(zip(*map(self._search_local_fit, range(self.n))))
        err2 = np.array(result[0]).reshape(-1, 1)
        hat = np.array(result[1]).reshape(-1, 1)
        aa = np.sum(err2 / ((1 - hat) ** 2))
        RSS = np.sum(err2)
        tr_S = np.sum(hat)
        llf = -np.log(RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2

        return CalAicObj(tr_S, float(llf), float(aa), self.n)

    def _search_local_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influx = np.dot(self.X[i], inv_xtx_xt[:, i])
        return reside * reside, influx

    def _local_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influx = np.dot(self.X[i], inv_xtx_xt[:, i])
        Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
        CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
        Si2 = np.sum(Si ** 2)
        return influx, reside, predict, betas.reshape(-1), CCT, Si2

    def _multi_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        pre = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - pre
        return betas.reshape(-1), pre, reside

    def cal_multi(self):
        """
        calculate betas, predict value and reside, use for searching best bandwidth in MGWR model by backfitting.
        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._multi_fit, range(self.n))))
        else:
            result = list(zip(*map(self._multi_fit, range(self.n))))
        betas = np.array(result[0])
        pre = np.array(result[1]).reshape(-1, 1)
        reside = np.array(result[2]).reshape(-1, 1)
        return CalMultiObj(betas, pre, reside)

    def fit(self):
        """
        To fit GWR model
        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._local_fit, range(self.n))))
        else:
            result = list(zip(*map(self._local_fit, range(self.n))))
        influ = np.array(result[0]).reshape(-1, 1)
        reside = np.array(result[1]).reshape(-1, 1)
        predict_value = np.array(result[2]).reshape(-1, 1)
        betas = np.array(result[3])
        CCT = np.array(result[4])
        tr_STS = np.array(result[5])
        return GWRResults(self.coords, self.X, self.y, self.bw, self.kernel, self.fixed,
                          influ, reside, predict_value, betas, CCT, tr_STS)


class MGWR(GWR):
    """
    Multiscale Geographically Weighted Regression
    """
    def __init__(
            self,
            coords: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            selector,
            kernel: str = 'bisquare',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        """
        Parameters
        ----------
        coords        : array-like
                        n*2, spatial coordinates of the observations, if it's latitude and longitude,
                        the first column should be longitude

        X             : array-like
                        n*k, independent variable, excluding the constant

        y             : array-like
                        n*1, dependent variable

        selector      :SearchMGWRParameter object
                       valid SearchMGWRParameter that has successfully called
                       the "search" method. This parameter passes on
                       information from GAM model estimation including optimal
                       bandwidths.

        kernel        : string
                        type of kernel function used to weight observations;
                        available options:
                        'gaussian'
                        'bisquare'
                        'exponential'

        fixed         : bool
                        True for distance based kernel function and  False for
                        adaptive (nearest neighbor) kernel function (default)

        constant      : bool
                        True to include intercept (default) in model and False to exclude
                        intercept.

        thread        : int
                        The number of processes in parallel computation. If you have a large amount of data,
                        you can use it

        convert       : bool
                        Whether to convert latitude and longitude to plane coordinates.
        Examples
        --------
        import numpy as np
        from mgtwr.sel import SearchMGWRParameter
        from mgtwr.model import MGWR
        np.random.seed(10)
        u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
        v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
        t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
        x1 = np.random.uniform(0, 1, (1728, 1))
        x2 = np.random.uniform(0, 1, (1728, 1))
        epsilon = np.random.randn(1728, 1)
        beta0 = 5
        beta1 = 3 + (u + v + t)/6
        beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
        y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
        coords = np.hstack([u, v])
        X = np.hstack([x1, x2])
        sel_multi = SearchMGWRParameter(coords, X, y, kernel='gaussian', fixed=True)
        bws = sel_multi.search(multi_bw_max=[40], verbose=True)
        mgwr = MGWR(coords, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
        print(mgwr.R2)
        0.7045642214972343
        """
        self.selector = selector
        self.bws = self.selector.bws[0]  # final set of bandwidth
        self.bws_history = selector.bws[1]  # bws history in back_fitting
        self.betas = selector.bws[3]
        bw_init = self.selector.bws[5]  # initialization bandwidth
        super().__init__(
            coords, X, y, bw_init, kernel=kernel, fixed=fixed, constant=constant, thread=thread, convert=convert)
        self.n_chunks = None
        self.ENP_j = None

    def _chunk_compute(self, chunk_id=0):
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k))  # partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  # n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                                                                                   chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT,

    def fit(self, n_chunks=1):
        """
        Compute MGWR inference by chunk to reduce memory footprint.
        """
        self.n_chunks = n_chunks
        pre = np.sum(self.X * self.betas, axis=1).reshape(-1, 1)
        result = map(self._chunk_compute, (range(n_chunks)))
        result_list = list(zip(*result))
        ENP_j = np.sum(np.array(result_list[0]), axis=0)
        CCT = np.sum(np.array(result_list[1]), axis=0)
        return MGWRResults(
            self.coords, self.X, self.y, self.bws, self.kernel, self.fixed,
            self.bws_history, self.betas, pre, ENP_j, CCT)


class GTWR(BaseModel):
    """
    Geographically and Temporally Weighted Regression

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observations

    t             : array-like
                    n*1, time location

    X             : array-like
                        n*k, independent variable, excluding the constant

    y             : array-like
                    n*1, dependent variable

    bw            : scalar
                    bandwidth value consisting of either a distance or N
                    nearest neighbors; user specified or obtained using
                    sel

    tau           : scalar
                    spatio-temporal scale

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : bool
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : bool
                    True to include intercept (default) in model and False to exclude
                    intercept.

    Examples
    --------
    import numpy as np
    from mgtwr.model import GTWR
    np.random.seed(10)
    u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i-1) % 144) // 12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t)/6
    beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
    gtwr = GTWR(coords, t, X, y, 0.8, 1.9, kernel='gaussian', fixed=True).fit()
    print(gtwr.R2)
    0.9899869616636376
    """

    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            t: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
            bw: float,
            tau: float,
            kernel: str = 'gaussian',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        super(GTWR, self).__init__(X, y, kernel, fixed, constant)
        if thread < 1 or not isinstance(thread, int):
            raise ValueError('thread should be an integer greater than or equal to 1')
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.t = t
        self.bw = bw
        self.tau = tau
        self.bw_s = self.bw
        self.bw_t = np.sqrt(self.bw ** 2 / self.tau)
        self.thread = thread

    def _build_wi(self, i, bw, tau):
        """
        calculate Weight matrix
        """
        try:
            gtwr_kernel = GTWRKernel(self.coords, self.t, bw, tau, fixed=self.fixed, function=self.kernel)
            distance = gtwr_kernel.cal_distance(i)
            wi = gtwr_kernel.cal_kernel(distance)
        except BaseException:
            raise  # TypeError('Unsupported kernel function  ', kernel)

        return wi

    def cal_aic(self):
        """
        use for calculating AICc, BIC, CV and so on.
        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._search_local_fit, range(self.n))))
        else:
            result = list(zip(*map(self._search_local_fit, range(self.n))))
        err2 = np.array(result[0]).reshape(-1, 1)
        hat = np.array(result[1]).reshape(-1, 1)
        aa = np.sum(err2 / ((1 - hat) ** 2))
        RSS = np.sum(err2)
        tr_S = np.sum(hat)
        llf = -np.log(RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2

        return CalAicObj(tr_S, float(llf), float(aa), self.n)

    def _search_local_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, xtx_inv_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influ = np.dot(self.X[i], xtx_inv_xt[:, i])
        return reside * reside, influ

    def _local_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, xtx_inv_xt = _compute_betas_gwr(self.y, self.X, wi)
        predict = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - predict
        influ = np.dot(self.X[i], xtx_inv_xt[:, i])
        Si = np.dot(self.X[i], xtx_inv_xt).reshape(-1)
        CCT = np.diag(np.dot(xtx_inv_xt, xtx_inv_xt.T)).reshape(-1)
        Si2 = np.sum(Si ** 2)
        return influ, reside, predict, betas.reshape(-1), CCT, Si2

    def _multi_fit(self, i):
        wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        pre = np.dot(self.X[i], betas)[0]
        reside = self.y[i] - pre
        return betas.reshape(-1), pre, reside

    def cal_multi(self):
        """
        calculate betas, predict value and reside, use for searching best bandwidth in MGWR model by backfitting.
        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._multi_fit, range(self.n))))
        else:
            result = list(zip(*map(self._multi_fit, range(self.n))))
        betas = np.array(result[0])
        pre = np.array(result[1]).reshape(-1, 1)
        reside = np.array(result[2]).reshape(-1, 1)
        return CalMultiObj(betas, pre, reside)

    def fit(self):
        """
        fit GTWR models

        """
        if self.thread > 1:
            pool = mp.Pool(self.thread)
            result = list(zip(*pool.map(self._local_fit, range(self.n))))
        else:
            result = list(zip(*map(self._local_fit, range(self.n))))
        influ = np.array(result[0]).reshape(-1, 1)
        reside = np.array(result[1]).reshape(-1, 1)
        predict_value = np.array(result[2]).reshape(-1, 1)
        betas = np.array(result[3])
        CCT = np.array(result[4])
        tr_STS = np.array(result[5])
        return GTWRResults(
            self.coords, self.t, self.X, self.y, self.bw, self.tau, self.kernel, self.fixed,
            influ, reside, predict_value, betas, CCT, tr_STS
        )


class MGTWR(GTWR):
    """
    Multiscale GTWR estimation and inference.

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observatons

    t             : array
                    n*1, time location

    X             : array-like
                        n*k, independent variable, excluding the constant

    y             : array-like
                    n*1, dependent variable

    selector      : SearchMGTWRParameter object
                    valid SearchMGTWRParameter object that has successfully called
                    the "search" method. This parameter passes on
                    information from GAM model estimation including optimal
                    bandwidths.

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : bool
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : bool
                    True to include intercept (default) in model and False to exclude
                    intercept.
    Examples
    --------
    import numpy as np
    from mgtwr.sel import SearchMGTWRParameter
    from mgtwr.model import MGTWR
    np.random.seed(10)
    u = np.array([(i-1) % 12 for i in range(1, 1729)]).reshape(-1, 1)
    v = np.array([((i-1) % 144)//12 for i in range(1, 1729)]).reshape(-1, 1)
    t = np.array([(i-1) // 144 for i in range(1, 1729)]).reshape(-1, 1)
    x1 = np.random.uniform(0, 1, (1728, 1))
    x2 = np.random.uniform(0, 1, (1728, 1))
    epsilon = np.random.randn(1728, 1)
    beta0 = 5
    beta1 = 3 + (u + v + t)/6
    beta2 = 3 + ((36-(6-u)**2)*(36-(6-v)**2)*(36-(6-t)**2)) / 128
    y = beta0 + beta1 * x1 + beta2 * x2 + epsilon
    coords = np.hstack([u, v])
    X = np.hstack([x1, x2])
    sel_multi = SearchMGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
    bws = sel_multi.search(multi_bw_min=[0.1], verbose=True, tol_multi=1.0e-4)
    mgtwr = MGTWR(coords, t, X, y, sel_multi, kernel='gaussian', fixed=True).fit()
    print(mgtwr.R2)
    0.9972924820674222
    """

    def __init__(
            self,
            coords: np.ndarray,
            t: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            selector,
            kernel: str = 'bisquare',
            fixed: bool = False,
            constant: bool = True,
            thread: int = 1,
            convert: bool = False
    ):
        self.selector = selector
        self.bws = self.selector.bws[0]  # final set of bandwidth
        self.taus = self.selector.bws[1]
        self.bw_ts = np.sqrt(self.bws ** 2 / self.taus)
        self.bws_history = selector.bws[2]  # bws history in back_fitting
        self.taus_history = selector.bws[3]
        self.betas = selector.bws[5]
        bw_init = self.selector.bws[7]  # initialization bandwidth
        tau_init = self.selector.bws[8]
        super().__init__(coords, t, X, y, bw_init, tau_init,
                         kernel=kernel, fixed=fixed, constant=constant, thread=thread, convert=convert)
        self.n_chunks = None
        self.ENP_j = None

    def _chunk_compute(self, chunk_id=0):
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index),
                       k))  # partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw, self.tau).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  # n by chunk_size

        for iter_i in range(self.bws_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(
                                                                                   chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_history[iter_i, j],
                                            self.taus_history[iter_i, j])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(
                axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT,

    def fit(self, n_chunks=1):
        """
        Compute MGTWR inference by chunk to reduce memory footprint.
        """
        self.n_chunks = n_chunks
        pre = np.sum(self.X * self.betas, axis=1).reshape(-1, 1)
        result = map(self._chunk_compute, (range(n_chunks)))
        result_list = list(zip(*result))
        ENP_j = np.sum(np.array(result_list[0]), axis=0)
        CCT = np.sum(np.array(result_list[1]), axis=0)
        return MGTWRResults(
            self.coords, self.t, self.X, self.y, self.bws, self.taus, self.kernel, self.fixed, self.bw_ts,
            self.bws_history, self.taus_history, self.betas, pre, ENP_j, CCT)
