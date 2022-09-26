import numpy as np
import pandas as pd
from typing import Union


class CalAicObj:

    def __init__(self, tr_S, llf, aa, n):
        self.tr_S = tr_S
        self.llf = llf
        self.aa = aa
        self.n = n


class CalMultiObj:

    def __init__(self, betas, pre, reside):
        self.betas = betas
        self.pre = pre
        self.reside = reside


class BaseModel:
    """
    Is the parent class of most models
    """
    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame, pd.Series],
            y: Union[np.ndarray, pd.DataFrame, pd.Series],
            kernel: str,
            fixed: bool,
            constant: bool,
    ):
        self.X = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        self.y = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else y
        if len(y.shape) > 1 and y.shape[1] != 1:
            raise ValueError('Label should be one-dimensional arrays')
        if len(y.shape) == 1:
            self.y = self.y.reshape(-1, 1)
        self.kernel = kernel
        self.fixed = fixed
        self.constant = constant
        self.n = X.shape[0]
        if self.constant:
            if len(self.X.shape) == 1 and np.all(self.X == 1):
                raise ValueError("You've already passed in a constant sequence, use constant=False instead")
            for j in range(self.X.shape[1]):
                if np.all(self.X[:, j] == 1):
                    raise ValueError("You've already passed in a constant sequence, use constant=False instead")
            self.X = np.hstack([np.ones((self.n, 1)), X])
        self.k = self.X.shape[1]


class Results(BaseModel):
    """
    Is the result parent class of all models
    """

    def __init__(
            self,
            X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            kernel: str,
            fixed: bool,
            influ: np.ndarray,
            reside,
            predict_value: np.ndarray,
            betas: np.ndarray,
            tr_STS: float
    ):
        super(Results, self).__init__(X, y, kernel, fixed, constant=False)
        self.influ = influ
        self.reside = reside
        self.predict_value = predict_value
        self.betas = betas
        self.tr_S = np.sum(influ)
        self.ENP = self.tr_S
        self.tr_STS = tr_STS
        self.TSS = np.sum((y - np.mean(y)) ** 2)
        self.RSS = np.sum(reside ** 2)
        self.sigma2 = self.RSS / (self.n - self.tr_S)
        self.std_res = self.reside / (np.sqrt(self.sigma2 * (1.0 - self.influ)))
        self.cooksD = self.std_res ** 2 * self.influ / (self.tr_S * (1.0 - self.influ))
        self.df_model = self.n - self.tr_S
        self.df_reside = self.n - 2.0 * self.tr_S + self.tr_STS
        self.R2 = 1 - self.RSS / self.TSS
        self.adj_R2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
        self.llf = -np.log(self.RSS) * self.n / 2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2
        self.aic = -2.0 * self.llf + 2.0 * (self.tr_S + 1)
        self.aicc = self.aic + 2.0 * self.tr_S * (self.tr_S + 1.0) / (self.n - self.tr_S - 1.0)
        self.bic = -2.0 * self.llf + (self.k + 1) * np.log(self.n)


class GWRResults(Results):

    def __init__(
            self, coords, X, y, bw, kernel, fixed, influ, reside, predict_value, betas, CCT, tr_STS
    ):
        """
        betas               : array
                              n*k, estimated coefficients

        predict             : array
                              n*1, predict y values

        CCT                 : array
                              n*k, scaled variance-covariance matrix

        df_model            : integer
                              model degrees of freedom

        df_reside           : integer
                              residual degrees of freedom

        reside              : array
                              n*1, residuals of the response

        RSS                 : scalar
                              residual sum of squares

        CCT                 : array
                              n*k, scaled variance-covariance matrix

        ENP                 : scalar
                              effective number of parameters, which depends on
                              sigma2

        tr_S                : float
                              trace of S (hat) matrix

        tr_STS              : float
                              trace of STS matrix

        R2                  : float
                              R-squared for the entire model (1- RSS/TSS)

        adj_R2              : float
                              adjusted R-squared for the entire model

        aic                 : float
                              Akaike information criterion

        aicc                : float
                              corrected Akaike information criterion
                              to account for model complexity (smaller
                              bandwidths)

        bic                 : float
                              Bayesian information criterion

        sigma2              : float
                              sigma squared (residual variance) that has been
                              corrected to account for the ENP

        std_res             : array
                              n*1, standardised residuals

        bse                 : array
                              n*k, standard errors of parameters (betas)

        influ               : array
                              n*1, leading diagonal of S matrix

        CooksD              : array
                              n*1, Cook's D

        tvalues             : array
                              n*k, local t-statistics

        llf                 : scalar
                              log-likelihood of the full model; see
                              pysal.contrib.glm.family for damily-sepcific
                              log-likelihoods
        """

        super(GWRResults, self).__init__(
            X, y, kernel, fixed, influ, reside, predict_value, betas, tr_STS)
        self.coords = coords
        self.bw = bw
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.betas / self.bse


class GTWRResults(Results):

    def __init__(
            self, coords, t, X, y, bw, tau, kernel, fixed, influ, reside, predict_value, betas, CCT, tr_STS
    ):
        """
        tau:        : scalar
                      spatio-temporal scale
        bw_s        : scalar
                      spatial bandwidth
        bw_t        : scalar
                      temporal bandwidth
        See Also GWRResults
        """

        super(GTWRResults, self).__init__(X, y, kernel, fixed, influ, reside, predict_value, betas, tr_STS)
        self.coords = coords
        self.t = t
        self.bw = bw
        self.tau = tau
        self.bw_s = self.bw
        self.bw_t = np.sqrt(self.bw ** 2 / self.tau)
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.betas / self.bse


class MGWRResults(BaseModel):

    def __init__(self, coords, X, y, bws, kernel, fixed, bws_history, betas,
                 predict_value, ENP_j, CCT):
        """
        bws         : array-like
                      corresponding spatial bandwidth of all variables
        ENP_j       : array-like
                      effective number of paramters, which depends on
                      sigma2, for each covariate in the model

        See Also GWRResults
        """
        super(MGWRResults, self).__init__(X, y, kernel, fixed, constant=False)
        self.coords = coords
        self.bws = bws
        self.bws_history = bws_history
        self.predict_value = predict_value
        self.betas = betas
        self.ENP_j = ENP_j
        self.reside = self.y - self.predict_value
        self.tr_S = np.sum(self.ENP_j)
        self.ENP = self.tr_S
        self.TSS = np.sum((self.y - np.mean(self.y)) ** 2)
        self.RSS = np.sum(self.reside ** 2)
        self.sigma2 = self.RSS / (self.n - self.tr_S)
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.t_values = self.betas / self.bse
        self.df_model = self.n - self.tr_S
        self.R2 = 1 - self.RSS / self.TSS
        self.adj_R2 = 1 - (1 - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
        self.llf = -np.log(self.RSS) * self.n / \
                   2 - (1 + np.log(np.pi / self.n * 2)) * self.n / 2
        self.aic = -2.0 * self.llf + 2.0 * (self.tr_S + 1)
        self.aic_c = self.aic + 2.0 * self.tr_S * (self.tr_S + 1.0) / \
                     (self.n - self.tr_S - 1.0)
        self.bic = -2.0 * self.llf + (self.k + 1) * np.log(self.n)


class MGTWRResults(MGWRResults):

    def __init__(self, coords, t, X, y, bws, taus, kernel, fixed, bw_ts, bws_history, taus_history, betas,
                 predict_value, ENP_j, CCT):
        """
        taus        : array-like
                     corresponding spatio-temporal scale of all variables
        bws         : array-like
                     corresponding spatio bandwidth of all variables
        bw_ts       : array-like
                     corresponding temporal bandwidth of all variables
        See Also
        -------------
        MGWRResults
        GWRResults
        """
        super(MGTWRResults, self).__init__(
            coords, X, y, bws, kernel, fixed, bws_history, betas, predict_value, ENP_j, CCT)
        self.t = t
        self.taus = taus
        self.bw_ts = bw_ts
        self.taus_history = taus_history
