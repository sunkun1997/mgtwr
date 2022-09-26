import numpy as np
from typing import Union
import pandas as pd
from .diagnosis import get_AICc, get_AIC, get_BIC, get_CV
from .obj import BaseModel
from scipy.spatial.distance import pdist
from .model import GWR, GTWR
from .function import golden_section, surface_to_plane, print_time, twostep_golden_section, multi_bw, multi_bws

getDiag = {'AICc': get_AICc, 'AIC': get_AIC, 'BIC': get_BIC, 'CV': get_CV}

delta = 0.38197


class SearchGWRParameter(BaseModel):
    """
    Select bandwidth for GWR model

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observations

    y             : array-like
                    n*1, dependent variable

    X             : array-like
                    n*k, independent variable, excluding the constant

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept.

    Examples
    --------
    import numpy as np
    from mgtwr.sel import SearchGWRParameter
    np.random.seed(1)
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
    sel = SearchGWRParameter(coords, X, y, kernel='gaussian', fixed=True)
    bw = sel.search(bw_max=40, verbose=True)
    2.0
    """

    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
            kernel: str = 'exponential',
            fixed: bool = False,
            constant: bool = True,
            convert: bool = False,
            thread: int = 1
    ):

        super(SearchGWRParameter, self).__init__(X, y, kernel, fixed, constant)
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.int_score = not self.fixed
        self.thread = thread

    @print_time
    def search(self,
               criterion: str = 'AICc',
               bw_min: float = None,
               bw_max: float = None,
               tol: float = 1.0e-6,
               bw_decimal: int = 0,
               max_iter: int = 200,
               verbose: bool = True,
               time_cost: bool = False
               ):
        """
        Method to select one unique bandwidth for a GWR model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol

        bw_decimal      : scalar
                         The number of bandwidth's decimal places saved during the search

        verbose        : bool
                         If true, bandwidth searching history is printed out; default is False.
        time_cost      : bool
                         If true, print run time
        """

        def gwr_func(x):
            return getDiag[criterion](GWR(
                self.coords, self.X, self.y, x, kernel=self.kernel,
                fixed=self.fixed, constant=False, thread=self.thread).cal_aic())

        bw_min, bw_max = self._init_section(bw_min, bw_max)
        bw = golden_section(bw_min, bw_max, delta, bw_decimal, gwr_func, tol, max_iter, verbose)
        return bw

    def _init_section(self, bw_min, bw_max):
        if bw_min is not None and bw_max is not None:
            return bw_min, bw_max

        if len(self.X) > 0:
            n_glob = self.X.shape[1]
        else:
            n_glob = 0
        if self.constant:
            n_vars = n_glob + 1
        else:
            n_vars = n_glob
        n = np.array(self.coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            sq_dists = pdist(self.coords)
            a = np.min(sq_dists) / 2.0
            c = np.max(sq_dists)
        if bw_min is None:
            bw_min = a
        if bw_max is None:
            bw_max = c

        return bw_min, bw_max


class SearchMGWRParameter(BaseModel):

    def __init__(
            self,
            coords: Union[np.ndarray, pd.DataFrame],
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.DataFrame],
            kernel: str = 'exponential',
            fixed: bool = False,
            constant: bool = True,
            convert: bool = False,
            thread: int = 1
    ):

        super(SearchMGWRParameter, self).__init__(X, y, kernel, fixed, constant)
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.int_score = not self.fixed
        self.thread = thread
        self.criterion = None
        self.bws = None
        self.tol = None
        self.bw_decimal = None

    @print_time
    def search(
            self,
            criterion: str = 'AICc',
            bw_min: float = None,
            bw_max: float = None,
            tol: float = 1.0e-6,
            bw_decimal: int = 1,
            init_bw: float = None,
            multi_bw_min: list = None,
            multi_bw_max: list = None,
            tol_multi: float = 1.0e-5,
            bws_same_times: int = 5,
            verbose: bool = False,
            rss_score: bool = False,
            time_cost: bool = False
            ):
        """
        Method to select one unique bandwidth and Spatio-temporal scale for a gtwr model or a
        bandwidth vector and Spatio-temporal scale vector for a mgwr model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        multi_bw_min   : list
                         min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept
        multi_bw_max   : list
                         max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        tol            : float
                         tolerance used to determine convergence
        bw_decimal     : int
                        The number of bw decimal places reserved
        init_bw        : float
                         None (default) to initialize MGTWR with a bandwidth
                         derived from GTWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.
        tol_multi      : convergence tolerance for the multiple bandwidth
                         back fitting algorithm; a larger tolerance may stop the
                         algorithm faster though it may result in a less optimal
                         model
        bws_same_times : If bandwidths keep the same between iterations for
                         bws_same_times (default 5) in backfitting, then use the
                         current set of bandwidths as final bandwidths.
        rss_score      : True to use the residual sum of squares to evaluate
                         each iteration of the multiple bandwidth back fitting
                         routine and False to use a smooth function; default is
                         False
        verbose        : Boolean
                         If true, bandwidth searching history is printed out; default is False.
        time_cost      : bool
                        If true, print run time
        """
        self.criterion = criterion
        self.tol = tol
        self.bw_decimal = bw_decimal
        if multi_bw_min is not None:
            if len(multi_bw_min) == self.k:
                multi_bw_min = multi_bw_min
            elif len(multi_bw_min) == 1:
                multi_bw_min = multi_bw_min * self.k
            else:
                raise AttributeError(
                    "multi_bw_min must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")
        else:
            a = self._init_section(bw_min, bw_max)[0]
            multi_bw_min = [a] * self.k

        if multi_bw_max is not None:
            if len(multi_bw_max) == self.k:
                multi_bw_max = multi_bw_max
            elif len(multi_bw_max) == 1:
                multi_bw_max = multi_bw_max * self.k
            else:
                raise AttributeError(
                    "multi_bw_max must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")
        else:
            c = self._init_section(bw_min, bw_max)[1]
            multi_bw_max = [c] * self.k

        self.bws = multi_bw(init_bw, self.X, self.y, self.n, self.k, tol_multi,
                            rss_score, self.gwr_func, self.bw_func, self.sel_func, multi_bw_min, multi_bw_max,
                            bws_same_times, verbose=verbose)
        return self.bws

    def gwr_func(self, X, y, bw):
        res = GWR(self.coords, X, y, bw, kernel=self.kernel,
                  fixed=self.fixed, constant=False, thread=self.thread).cal_multi()
        return res

    def bw_func(self, X, y):
        selector = SearchGWRParameter(self.coords, X, y, kernel=self.kernel, fixed=self.fixed,
                                      constant=False, thread=self.thread)
        return selector

    def sel_func(self, bw_func, bw_min=None, bw_max=None):
        return bw_func.search(criterion=self.criterion, bw_min=bw_min, bw_max=bw_max,
                              tol=self.tol, bw_decimal=self.bw_decimal, verbose=False)

    def _init_section(self, bw_min, bw_max):

        a = bw_min if bw_min is not None else 0
        if bw_max is not None:
            c = bw_max
        else:
            c = max(np.max(self.coords[:, 0]) - np.min(self.coords[:, 0]),
                    np.max(self.coords[:, 1]) - np.min(self.coords[:, 1]))
        return a, c


class SearchGTWRParameter(BaseModel):
    """
    Select bandwidth for GTWR model

    Parameters
    ----------
    coords        : array-like
                    n*2, collection of n sets of (x,y) coordinates of
                    observations

    t             : array-like
                    n*1, time location

    y             : array-like
                    n*1, dependent variable

    X             : array-like
                    n*k, independent variable, excluding the constant

    kernel        : string
                    type of kernel function used to weight observations;
                    available options:
                    'gaussian'
                    'bisquare'
                    'exponential'

    fixed         : boolean
                    True for distance based kernel function and  False for
                    adaptive (nearest neighbor) kernel function (default)

    constant      : boolean
                    True to include intercept (default) in model and False to exclude
                    intercept.

    Examples
    --------
    import numpy as np
    from mgtwr.sel import SearchGTWRParameter
    np.random.seed(1)
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
    sel = SearchGTWRParameter(coords, t, X, y, kernel='gaussian', fixed=True)
    bw, tau = sel.search(tau_max=20, verbose=True)
    0.9, 1.5
    """

    def __init__(
            self,
            coords: np.ndarray,
            t: np.ndarray,
            X: np.ndarray,
            y: np.ndarray,
            kernel: str = 'exponential',
            fixed: bool = False,
            constant: bool = True,
            convert: bool = False,
            thread: int = 1
    ):

        super(SearchGTWRParameter, self).__init__(X, y, kernel, fixed, constant)
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.t = t
        self.int_score = not self.fixed
        self.thread = thread

    @print_time
    def search(
            self,
            criterion: str = 'AICc',
            bw_min: float = None,
            bw_max: float = None,
            tau_min: float = None,
            tau_max: float = None,
            tol: float = 1.0e-6,
            bw_decimal: int = 1,
            tau_decimal: int = 1,
            max_iter: int = 200,
            verbose: bool = False,
            time_cost: bool = False
            ):
        """
        Method to select one unique bandwidth and Spatio-temporal scale for a GTWR model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        tau_min        : float
                         min value used in spatio-temporal scale search
        tau_max        : float
                         max value used in spatio-temporal scale search
        tol            : float
                         tolerance used to determine convergence
        max_iter       : integer
                         max iterations if no convergence to tol
        bw_decimal      : scalar
                         The number of bandwidth's decimal places saved during the search
        tau_decimal     : scalar
                         The number of Spatio-temporal decimal places saved during the search
        verbose        : Boolean
                         If true, bandwidth searching history is printed out; default is False.
        time_cost      : bool
                        If true, print run time
        """

        def gtwr_func(x, y):
            return getDiag[criterion](GTWR(
                self.coords, self.t, self.X, self.y, x, y, kernel=self.kernel,
                fixed=self.fixed, constant=False, thread=self.thread).cal_aic())

        bw_min, bw_max, tau_min, tau_max = self._init_section(bw_min, bw_max, tau_min, tau_max)
        bw, tau = twostep_golden_section(bw_min, bw_max, tau_min, tau_max, delta, gtwr_func, tol, max_iter, bw_decimal,
                                         tau_decimal, verbose)

        return bw, tau

    def _init_section(self, bw_min, bw_max, tau_min, tau_max):
        if (bw_min is not None) and (bw_max is not None) and (tau_min is not None) and (tau_max is not None):
            return bw_min, bw_max, tau_min, tau_max
        if len(self.X) > 0:
            n_glob = self.X.shape[1]
        else:
            n_glob = 0
        if self.constant:
            n_vars = n_glob + 1
        else:
            n_vars = n_glob
        n = np.array(self.coords).shape[0]

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            sq_dists = pdist(self.coords)
            a = np.min(sq_dists) / 2.0
            c = np.max(sq_dists)

        if bw_min is None:
            bw_min = a
        if bw_max is None:
            bw_max = c

        if tau_min is None:
            tau_min = 0
        if tau_max is None:
            tau_max = 4
        return bw_min, bw_max, tau_min, tau_max


class SearchMGTWRParameter(BaseModel):
    """
    Select bandwidth for MGTWR model

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
            kernel: str = 'exponential',
            fixed: bool = False,
            constant: bool = True,
            convert: bool = False,
            thread: int = 1
    ):

        super(SearchMGTWRParameter, self).__init__(X, y, kernel, fixed, constant)
        if isinstance(coords, pd.DataFrame):
            coords = coords.values
        self.coords = coords
        if convert:
            longitude = coords[:, 0]
            latitude = coords[:, 1]
            longitude, latitude = surface_to_plane(longitude, latitude)
            self.coords = np.hstack([longitude, latitude])
        self.t = t
        self.int_score = not self.fixed
        self.thread = thread
        self.criterion = None
        self.bws = None
        self.tol = None
        self.bw_decimal = None
        self.tau_decimal = None

    @print_time
    def search(
            self,
            criterion: str = 'AICc',
            bw_min: float = None,
            bw_max: float = None,
            tau_min: float = None,
            tau_max: float = None,
            tol: float = 1.0e-6,
            bw_decimal: int = 1,
            tau_decimal: int = 1,
            init_bw: float = None,
            init_tau: float = None,
            multi_bw_min: list = None,
            multi_bw_max: list = None,
            multi_tau_min: list = None,
            multi_tau_max: list = None,
            tol_multi: float = 1.0e-5,
            verbose: bool = False,
            rss_score: bool = False,
            time_cost: bool = False
            ):
        """
        Method to select one unique bandwidth and Spatio-temporal scale for a gtwr model or a
        bandwidth vector and Spatio-temporal scale vector for a mtgwr model.

        Parameters
        ----------
        criterion      : string
                         bw selection criterion: 'AICc', 'AIC', 'BIC', 'CV'
        bw_min         : float
                         min value used in bandwidth search
        bw_max         : float
                         max value used in bandwidth search
        tau_min        : float
                         min value used in spatio-temporal scale search
        tau_max        : float
                         max value used in spatio-temporal scale search
        multi_bw_min   : list
                         min values used for each covariate in mgwr bandwidth search.
                         Must be either a single value or have one value for
                         each covariate including the intercept
        multi_bw_max   : list
                         max values used for each covariate in mgwr bandwidth
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        multi_tau_min  : list
                         min values used for each covariate in mgtwr spatio-temporal scale
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        multi_tau_max  : max values used for each covariate in mgtwr spatio-temporal scale
                         search. Must be either a single value or have one value
                         for each covariate including the intercept
        tol            : float
                         tolerance used to determine convergence
        bw_decimal     : int
                        The number of bw decimal places reserved
        tau_decimal    : int
                        The number of tau decimal places reserved
        init_bw        : float
                         None (default) to initialize MGTWR with a bandwidth
                         derived from GTWR. Otherwise this option will choose the
                         bandwidth to initialize MGWR with.
        init_tau       : float
                         None (default) to initialize MGTWR with a spatio-temporal scale
                         derived from GTWR. Otherwise this option will choose the
                         spatio-temporal scale to initialize MGWR with.
        tol_multi      : convergence tolerance for the multiple bandwidth
                         back fitting algorithm; a larger tolerance may stop the
                         algorithm faster though it may result in a less optimal
                         model
        rss_score      : True to use the residual sum of squares to evaluate
                         each iteration of the multiple bandwidth back fitting
                         routine and False to use a smooth function; default is
                         False
        verbose        : Boolean
                         If true, bandwidth searching history is printed out; default is False.
        time_cost      : bool
                        If true, print run time
        """
        self.criterion = criterion
        self.tol = tol
        self.bw_decimal = bw_decimal
        self.tau_decimal = tau_decimal
        if multi_bw_min is not None:
            if len(multi_bw_min) == self.k:
                multi_bw_min = multi_bw_min
            elif len(multi_bw_min) == 1:
                multi_bw_min = multi_bw_min * self.k
            else:
                raise AttributeError(
                    "multi_bw_min must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")
        else:
            a = self._init_section(bw_min, bw_max, tau_min, tau_max)[0]
            multi_bw_min = [a] * self.k

        if multi_bw_max is not None:
            if len(multi_bw_max) == self.k:
                multi_bw_max = multi_bw_max
            elif len(multi_bw_max) == 1:
                multi_bw_max = multi_bw_max * self.k
            else:
                raise AttributeError(
                    "multi_bw_max must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")
        else:
            c = self._init_section(bw_min, bw_max, tau_min, tau_max)[1]
            multi_bw_max = [c] * self.k

        if multi_tau_min is not None:
            if len(multi_tau_min) == self.k:
                multi_tau_min = multi_tau_min
            elif len(multi_tau_min) == 1:
                multi_tau_min = multi_tau_min * self.k
            else:
                raise AttributeError(
                    "multi_tau_min must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " variates including the intercept")
        else:
            A = self._init_section(bw_min, bw_max, tau_min, tau_max)[2]
            multi_tau_min = [A] * self.k

        if multi_tau_max is not None:
            if len(multi_tau_max) == self.k:
                multi_tau_max = multi_tau_max
            elif len(multi_tau_max) == 1:
                multi_tau_max = multi_tau_max * self.k
            else:
                raise AttributeError(
                    "multi_tau_max must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " variates including the intercept")
        else:
            C = self._init_section(bw_min, bw_max, tau_min, tau_max)[3]
            multi_tau_max = [C] * self.k

        self.bws = multi_bws(init_bw, init_tau, self.X, self.y, self.n, self.k, tol_multi,
                             rss_score, self.gtwr_func, self.bw_func, self.sel_func, multi_bw_min, multi_bw_max,
                             multi_tau_min, multi_tau_max, verbose=verbose)
        return self.bws

    def gtwr_func(self, X, y, bw, tau):
        return GTWR(self.coords, self.t, X, y, bw, tau, kernel=self.kernel,
                    fixed=self.fixed, constant=False, thread=self.thread).cal_multi()

    def bw_func(self, X, y):
        selector = SearchGTWRParameter(self.coords, self.t, X, y, kernel=self.kernel, fixed=self.fixed,
                                       constant=False, thread=self.thread)
        return selector

    def sel_func(self, bw_func, bw_min=None, bw_max=None, tau_min=None, tau_max=None):
        return bw_func.search(criterion=self.criterion, bw_min=bw_min, bw_max=bw_max, tau_min=tau_min, tau_max=tau_max,
                              tol=self.tol, bw_decimal=self.bw_decimal, tau_decimal=self.tau_decimal, verbose=False)

    def _init_section(self, bw_min, bw_max, tau_min, tau_max):

        a = bw_min if bw_min is not None else 0
        if bw_max is not None:
            c = bw_max
        else:
            c = max(np.max(self.coords[:, 0]) - np.min(self.coords[:, 0]),
                    np.max(self.coords[:, 1]) - np.min(self.coords[:, 1]))

        A = tau_min if tau_min is not None else 0
        C = tau_max if tau_max is not None else 4

        return a, c, A, C
