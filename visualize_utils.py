import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def OF_dynamics_log(t, c, a):
    """ Linear fitting curve

    params
    -----------------------
    t : array (1xT-dimensional) or scalar
        table/scalar containing timepoint(s)
    b, c : scalar
        model parameters to be fitted

    returns
    -----------------------
    function : scalar
        value of function at timepoints t with parameters a, b and c.

    """

    y = c + a * np.log(t)

    return y

def OF_dynamics_linear(t, c, a):
    """ Linear fitting curve

    params
    -----------------------
    t : array (1xT-dimensional) or scalar
        table/scalar containing timepoint(s)
    b, c : scalar
        model parameters to be fitted

    returns
    -----------------------
    function : scalar
        value of function at timepoints t with parameters a, b and c.

    """

    y = c + a * t

    return y

def r_squared(data, fit):
    """ Computes r-square which represents the proportion of the variance for a
    dependent variable that's explained by an independent variable.

    params
    -----------------------
    data : array dim(1, n)
        data with n timepoints
    fit : array dim(1, n)
        simulation of the model with n timepoints

    returns
    -----------------------
    r_squared: float
        value of the r-square

    """

    # average
    mean = np.nanmean(data)

    # compute residual sum of squares
    SS_res = np.nansum((data-fit)**2)

    # compute total sum of squares
    SS_tot = np.nansum((data-mean)**2)

    # coefficient of determination
    try:
        r_squared = 1 - SS_res/SS_tot
    except:
        r_squared = np.nan

    return r_squared