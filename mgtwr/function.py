import numpy as np
from scipy import linalg
import time
from typing import Callable
from copy import deepcopy


def print_time(func: Callable):
    def inner(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        m, s = divmod(end - start, 60)
        h, m = divmod(m, 60)
        if 'time_cost' in kwargs and kwargs['time_cost']:
            print("time cost: %d:%02d:%s" % (h, m, round(s, 3)))
        return res
    return inner


def _compute_betas_gwr(y, x, wi):
    """
    compute MLE coefficients using iwls routine

    Methods: p189, Iteratively (Re)weighted Least Squares (IWLS),
    Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002).
    Geographically weighted regression: the analysis of spatially varying relationships.
    """
    xt = (x * wi).T
    xtx = np.dot(xt, x)
    xtx_inv_xt = linalg.solve(xtx, xt)
    betas = np.dot(xtx_inv_xt, y)
    return betas, xtx_inv_xt


def surface_to_plane(
        longitude: np.ndarray,
        latitude: np.ndarray,
        central_longitude: int = 114
):

    r"""
    base on Gauss-Kruger projection

    equatorial radius: a = 6378136.49m

    polar radius: b = 6356755m

    so that

    first eccentricity :math:`e = \sqrt{a^2-b^2}/a`

    second eccentricity :math:`e' = \sqrt{a^2-b^2}/b`

    so that

    .. math::
        \begin{aligned}
            Y_{b0}=a^2B\beta_0/b +
            sin(B)\left(\beta_2cos(B)+\beta_4cos^3(B)+\beta_6cos^5(B)+\beta_8cos^7(B)\right)
        \end{aligned}
    where B is the latitude converted from degrees to radians and

    .. math::
        \begin{aligned}
            \beta_0 &= 1-\frac{3}{4}e'^2+\frac{45}{64}e'^4-\frac{175}{256}e'^6+
                       \frac{11025}{16384}e'^8 \\
            \beta_2 &= \beta_0 - 1 \\
            \beta_4 &= \frac{15}{32}e'^4-\frac{175}{384}e'^6+\frac{3675}{8192}e'^8 \\
            \beta_6 &= -\frac{35}{96}e'^6 + \frac{735}{2048}e'^8 \\
            \beta_8 &= \frac{315}{1024}e'^8 \\
        \end{aligned}

    so that the Y-axis is

    .. math::
        \begin{aligned}
            Y &= Y_{b0}+\frac{1}{2}Ntan(B)m^2+\frac{1}{24}\left(5-tan^2(B)+9\eta^2+4\eta^4
                \right)Ntan(B)m^4 \\
              &+ \frac{1}{720}\left(61-58tan^2(B)\right)Ntan(B)m^6
        \end{aligned}
    where L is the longitude subtracts the central longitude converted to radians and

    .. math::
        \begin{aligned}
            N &= a/\sqrt{1-(esin(B))^2} \\
            \eta &= e'cos(B) \\
            m &= Lcos(B) \\
        \end{aligned}
    so that the X_axis is

    .. math::
        \begin{aligned}
            X &= Nm+\frac{1}{6}\left(1-tan^2(B)+\eta^2\right)Nm^3 \\
              &+ \frac{1}{120}\left(5-18tan^2(B)+tan^4(B)+14\eta^2-58tan^2(B)\eta\right)Nm^5+500000
        \end{aligned}
    """
    a = 6378136.49
    b = 6356755

    e1 = np.sqrt(a ** 2 - b ** 2) / a
    e2 = np.sqrt(a ** 2 - b ** 2) / b
    beta0 = 1 - (3 / 4) * e2 ** 2 + (45 / 64) * e2 ** 4 - (175 / 256) * e2 ** 6 \
        + (11025 / 16384) * e2 ** 8
    beta2 = beta0 - 1
    beta4 = (15 / 32) * e2 ** 4 - (175 / 384) * e2 ** 6 + (3675 / 8192) * e2 ** 8
    beta6 = -(35 / 96) * e2 ** 6 + (735 / 2048) * e2 ** 8
    beta8 = (315 / 1024) * e2 ** 8

    L = np.radians(longitude - central_longitude)
    B = np.radians(latitude)
    cosB = np.cos(B)
    sinB = np.sin(B)
    tanB = np.tan(B)
    N = a / np.sqrt(1 - (e1 * sinB) ** 2)
    eta = e2 * cosB
    m = L * cosB
    Yb0 = a ** 2 * B * beta0 / b + sinB * \
        (beta2 * cosB + beta4 * cosB ** 3 + beta6 * cosB ** 5 + beta8 * cosB ** 7)
    Y = Yb0 + (1 / 2) * N * tanB * m ** 2 + (1 / 24) * (5 - tanB ** 2 + 9 * eta ** 2 + 4 * eta ** 4) * N * tanB * \
        m ** 4 + (1 / 720) * (61 - 58 * tanB ** 2) * N * tanB * m ** 6
    X = N * m + (1 / 6) * (1 - tanB ** 2 + eta ** 2) * N * m ** 3 + \
        (1 / 120) * (5 - 18 * tanB ** 2 + tanB ** 4 + 14 * eta ** 2 - 58 * tanB ** 2 * eta) * N * m ** 5 + 500000
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    return X, Y


def golden_section(a, c, delta, decimal, function, tol, max_iter, verbose=False):
    b = a + delta * np.abs(c - a)
    d = c - delta * np.abs(c - a)
    diff = 1.0e9
    iter_num = 0
    score_dict = {}
    opt_val = None
    while np.abs(diff) > tol and iter_num < max_iter:
        iter_num += 1
        b = np.round(b, decimal)
        d = np.round(d, decimal)

        if b in score_dict:
            score_b = score_dict[b]
        else:
            score_b = function(b)
            score_dict[b] = score_b

        if d in score_dict:
            score_d = score_dict[d]
        else:
            score_d = function(d)
            score_dict[d] = score_d

        if score_b <= score_d:
            opt_val = b
            opt_score = score_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)

        else:
            opt_val = d
            opt_score = score_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)

        opt_val = np.round(opt_val, decimal)
        diff = score_b - score_d
        if verbose:
            print('bw:', opt_val, ', score:', np.round(opt_score, 2))

    return opt_val


def onestep_golden_section(A, C, x, delta, tau_decimal, function, tol):
    iter_num = 0
    score_dict = {}
    diff = 1e9
    opt_score = None
    opt_tau = None
    B = A + delta * np.abs(C - A)
    D = C - delta * np.abs(C - A)
    while np.abs(diff) > tol and iter_num < 200:
        iter_num += 1
        B = np.round(B, tau_decimal)
        D = np.round(D, tau_decimal)
        if B in score_dict:
            score_B = score_dict[B]
        else:
            score_B = function(x, B)
            score_dict[B] = score_B

        if D in score_dict:
            score_D = score_dict[D]
        else:
            score_D = function(x, D)
            score_dict[D] = score_D
        if score_B <= score_D:
            opt_score = score_B
            opt_tau = B
            C = D
            D = B
            B = A + delta * np.abs(C - A)
        else:
            opt_score = score_D
            opt_tau = D
            A = B
            B = D
            D = C - delta * np.abs(C - A)
        diff = score_B - score_D
    return opt_tau, opt_score


def twostep_golden_section(
        a, c, A, C, delta, function,
        tol, max_iter, bw_decimal, tau_decimal, verbose=False):
    b = a + delta * np.abs(c - a)
    d = c - delta * np.abs(c - a)
    opt_bw = None
    opt_tau = None
    diff = 1e9
    score_dict = {}
    iter_num = 0
    while np.abs(diff) > tol and iter_num < max_iter:
        iter_num += 1
        b = np.round(b, bw_decimal)
        d = np.round(d, bw_decimal)
        if b in score_dict:
            tau_b, score_b = score_dict[b]
        else:
            tau_b, score_b = onestep_golden_section(A, C, b, delta, tau_decimal, function, tol)
            score_dict[b] = [tau_b, score_b]
        if d in score_dict:
            tau_d, score_d = score_dict[d]
        else:
            tau_d, score_d = onestep_golden_section(A, C, d, delta, tau_decimal, function, tol)
            score_dict[d] = [tau_d, score_d]

        if score_b <= score_d:
            opt_score = score_b
            opt_bw = b
            opt_tau = tau_b
            c = d
            d = b
            b = a + delta * np.abs(c - a)
        else:
            opt_score = score_d
            opt_bw = d
            opt_tau = tau_d
            a = b
            b = d
            d = c - delta * np.abs(c - a)
        diff = score_b - score_d
        if verbose:
            print('bw: ', opt_bw, ', tau: ', opt_tau, ', score: ', opt_score)
    return opt_bw, opt_tau


def multi_bw(init, X, y, n, k, tol, rss_score, gwr_func,
             bw_func, sel_func, multi_bw_min, multi_bw_max, bws_same_times,
             verbose=False):
    """
    Multiscale GWR bandwidth search procedure using iterative GAM backfitting
    """
    if init is None:
        bw = sel_func(bw_func(X, y))
        optim_model = gwr_func(X, y, bw)
    else:
        bw = init
        optim_model = gwr_func(X, y, init)
    bw_gwr = bw
    err = optim_model.reside
    betas = optim_model.betas
    XB = np.multiply(betas, X)
    rss = np.sum(err ** 2) if rss_score else None
    scores = []
    BWs = []
    bw_stable_counter = 0
    bws = np.empty(k)
    Betas = None

    for iters in range(1, 201):
        new_XB = np.zeros_like(X)
        Betas = np.zeros_like(X)

        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1))
            temp_y = temp_y + err
            temp_X = X[:, j].reshape((-1, 1))
            bw_class = bw_func(temp_X, temp_y)

            if bw_stable_counter >= bws_same_times:
                # If in backfitting, all bws not changing in bws_same_times (default 5) iterations
                bw = bws[j]
            else:
                bw = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j])

            optim_model = gwr_func(temp_X, temp_y, bw)
            err = optim_model.reside
            betas = optim_model.betas
            new_XB[:, j] = optim_model.pre.reshape(-1)
            Betas[:, j] = betas.reshape(-1)
            bws[j] = bw

        # If bws remain the same as from previous iteration
        if (iters > 1) and np.all(BWs[-1] == bws):
            bw_stable_counter += 1
        else:
            bw_stable_counter = 0

        num = np.sum((new_XB - XB) ** 2) / n
        den = np.sum(np.sum(new_XB, axis=1) ** 2)
        score = (num / den) ** 0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(betas, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy) ** 2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))

        if verbose:
            print("Current iteration:", iters, ",SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws]))

        if delta < tol:
            break

    opt_bw = BWs[-1]
    return opt_bw, np.array(BWs), np.array(scores), Betas, err, bw_gwr


def multi_bws(init_bw, init_tau, X, y, n, k, tol, rss_score,
              gtwr_func, bw_func, sel_func, multi_bw_min, multi_bw_max,
              multi_tau_min, multi_tau_max, verbose=False):
    """
    Multiscale GTWR bandwidth search procedure using iterative GAM back fitting
    """
    if init_bw or init_tau is None:
        bw, tau = sel_func(bw_func(X, y))
    else:
        bw, tau = init_bw, init_tau
    opt_model = gtwr_func(X, y, bw, tau)
    bw_gtwr = bw
    tau_gtwr = tau
    err = opt_model.reside
    betas = opt_model.betas

    XB = np.multiply(betas, X)
    rss = np.sum(err ** 2) if rss_score else None
    scores = []
    bws = np.empty(k)
    taus = np.empty(k)
    BWs = []
    Taus = []
    Betas = None

    for iter_num in range(1, 201):
        new_XB = np.zeros_like(X)
        Betas = np.zeros_like(X)

        for j in range(k):
            temp_y = XB[:, j].reshape((-1, 1))
            temp_y = temp_y + err
            temp_X = X[:, j].reshape((-1, 1))
            bw_class = bw_func(temp_X, temp_y)

            bw, tau = sel_func(bw_class, multi_bw_min[j], multi_bw_max[j],
                               multi_tau_min[j], multi_tau_max[j])

            opt_model = gtwr_func(temp_X, temp_y, bw, tau)
            err = opt_model.reside
            betas = opt_model.betas
            new_XB[:, j] = (betas * temp_X).reshape(-1)
            Betas[:, j] = betas.reshape(-1)
            bws[j] = bw
            taus[j] = tau

        num = np.sum((new_XB - XB) ** 2) / n
        den = np.sum(np.sum(new_XB, axis=1) ** 2)
        score = (num / den) ** 0.5
        XB = new_XB

        if rss_score:
            predy = np.sum(np.multiply(betas, X), axis=1).reshape((-1, 1))
            new_rss = np.sum((y - predy) ** 2)
            score = np.abs((new_rss - rss) / new_rss)
            rss = new_rss
        scores.append(deepcopy(score))
        delta = score
        BWs.append(deepcopy(bws))
        Taus.append(deepcopy(taus))

        if verbose:
            print("Current iteration:", iter_num, ",SOC:", np.round(score, 7))
            print("Bandwidths:", ', '.join([str(bw) for bw in bws]))
            print("taus:", ','.join([str(tau) for tau in taus]))

        if delta < tol:
            break
    opt_bws = BWs[-1]
    opt_tau = Taus[-1]
    return (opt_bws, opt_tau, np.array(BWs), np.array(Taus), np.array(scores),
            Betas, err, bw_gtwr, tau_gtwr)
