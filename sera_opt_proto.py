# %%
import torch
from torch import nn

import torch.optim as optim


# %%
import ctypes
from itertools import chain
import sys
import os
import numpy as np
import pandas as pd
from numpy import double
from numpy.ctypeslib import ndpointer
from sklearn.model_selection import train_test_split
from iron import adjbox
import platform

from ctypes import *

import matplotlib.pyplot as plt

phiMethods = ["extremes", "range"]


def phi_control(
    y, phi_parms=None, method=phiMethods, extr_type=None, control_pts=None, asym=True
):

    if phi_parms is not None:
        method = phi_parms["method"]
        extr_type = phi_parms["extr_type"]
        control_pts = phi_parms["control_pts"]

    if method == "range":
        control_pts = phi_range(y, control_pts=control_pts)
    else:
        # create phi_extreme and control the parameters
        control_pts = phi_extremes(y, extr_type=extr_type, asym=asym)
        method = "extremes"
    phiP = {
        "method": method,
        "npts": control_pts["npts"],
        "control_pts": control_pts["control_pts"],
    }

    return phiP


# Auxiliary function
def minmax(val_list):
    min_val = min(val_list)
    max_val = max(val_list)

    return (min_val, max_val)


def phi_extremes(y, extr_type="both", coef=1.5, asym=True):

    control_pts = []
    npts = None
    if asym:

        y = y.to_list()
        extr = adjbox.adjboxStats(y, coef=coef)

        r = minmax(y)
        if extr_type is None:
            extr_type = "both"
        if extr_type in ("both", "low"):
            ## adjL

            control_pts.append((extr["fence"][0], 1, 0))

        else:
            ## min
            control_pts.append((r[0], 0, 0))

        ## median
        control_pts.append((extr["stats"][2], 0, 0))

        if extr_type in ("both", "high"):

            ## adjH
            control_pts.append((extr["fence"][1], 1, 0))
        else:
            ## max
            control_pts.append((r[1], 0, 0))

        npts = len(control_pts)

    else:

        extr = boxplot(y)

        r = minmax(y)

        if (extr_type in ("both", "low")) & (any(x < extr[0][1] for x in extr[3])):

            ## adjL
            control_pts.append((extr[0][0], 1, 0))
        else:
            ## min
            control_pts.append((r[0], 0, 0))

        ## median
        control_pts.append((extr[0][2], 0, 0))

        if (extr_type in ("both", "high")) & (any(x > extr[0][4] for x in extr[3])):

            ## adjH
            control_pts.append((extr[0][4], 1, 0))
        else:
            ## max
            control_pts.append((r[1], 0, 0))

            npts = len(control_pts)

    latten_list = list(chain.from_iterable(control_pts))
    return {"npts": npts, "control_pts": latten_list}


def phi_range(y, control_pts):

    if type(control_pts) is dict:
        control_pts = np.reshape(
            np.array(control_pts["control_pts"]),
            (
                control_pts["npts"],
                int(len(control_pts["control_pts"]) / control_pts["npts"]),
            ),
        )

    if (
        (type(control_pts) is not np.ndarray)
        or (control_pts is None)
        or (np.shape(control_pts)[1] > 3)
        or (np.shape(control_pts)[1] < 2)
    ):
        #    sys.exit('The control.pts must be given as a matrix in the form: \n < x, y, m > or, alternatively, < x, y >')
        raise ValueError(
            "The control.pts must be given as a matrix in the form: \n < x, y, m > or, alternatively, < x, y >"
        )
    npts = len(control_pts)
    dx = control_pts[1:, 0] - control_pts[0 : (npts - 1), 0]

    if (None in dx) or (0 in dx):
        # sys.exit("'x' must be *strictly* increasing (non - NA)")
        raise ValueError("'x' must be *strictly* increasing (non - NA)")

    if (any(x > 1 for x in control_pts[:, 1])) or any(x < 0 for x in control_pts[:, 1]):
        # sys.exit("phi relevance function maps values only in [0,1]")
        raise ValueError("phi relevance function maps values only in [0,1]")

    control_pts = control_pts[np.argsort(control_pts[:, 0])]

    if np.shape(control_pts)[1] == 2:
        ## based on "monoH.FC" method
        dx = control_pts[1:, 0] - control_pts[0 : (npts - 1), 0]
        dy = control_pts[1:, 1] - control_pts[0 : (npts - 1), 1]
        Sx = dy / dx
        m = (
            Sx[1:]
            + Sx[
                0 : (npts - 2),
            ]
        ) / 2
        m = np.reshape(m, (len(m), 1))
        m = np.insert(m, (0, len(m)), 0, axis=0)
        control_pts = np.append(control_pts, m, axis=1)

    r = minmax(y)
    npts = np.shape(control_pts)[0]
    latten_list = list(chain.from_iterable(control_pts))

    return {"npts": npts, "control_pts": latten_list}


# Auxiliary function
def phi2double(phi_parms):

    phi_parms_double = []
    if phi_parms["method"] == "extremes":
        phi_parms_double.append(0)
    elif phi_parms["method"] == "range":
        phi_parms_double.append(1)

    phi_parms_double.append(double(phi_parms["npts"]))
    phi_parms_double = np.append(phi_parms_double, phi_parms["control_pts"])

    return phi_parms_double


def phi(y, phi_parms=None, only_phi=True):
    if phi_parms is None:
        phi_parms = phi_control(y)
    n = len(y)

    if sys.platform == "win32":
        if platform.architecture()[0] == "64bit":
            dir = os.path.dirname(sys.modules["iron"].__file__)
            path = os.path.join(dir, "phi64.dll")
            phi_c = cdll.LoadLibrary(path)
        else:
            dir = os.path.dirname(sys.modules["iron"].__file__)
            path = os.path.join(dir, "phi.dll")
            phi_c = cdll.LoadLibrary(path)
    elif sys.platform == "darwin":
        dir = os.path.dirname(sys.modules["iron"].__file__)
        path = os.path.join(dir, "phi_mac.so")
        phi_c = cdll.LoadLibrary(path)
    elif sys.platform == "linux":
        dir = os.path.dirname(sys.modules["iron"].__file__)
        path = os.path.join(dir, "phi_linux.so")
        phi_c = cdll.LoadLibrary(path)

    py2phi = phi_c.py2phi
    py2phi.restype = None
    py2phi.argtypes = [
        ctypes.c_size_t,
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ]

    y_phi_all = np.empty((3 * n))
    py2phi(n, y.values, phi2double(phi_parms), y_phi_all)
    phis = {
        "y_phi": y_phi_all[0:n],
        "yd_phi": y_phi_all[n : 2 * n],
        "ydd_phi": y_phi_all[2 * n : 3 * n],
    }

    if only_phi:
        return phis["y_phi"]
    else:
        return phis


# Auxiliary function
def boxplot(x):
    median = np.median(x)
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = x[x <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = x[x >= lower_quartile - 1.5 * iqr].min()
    return {
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "median": median,
    }


def sera(
    trues,
    preds,
    phi_trues=None,
    ph=None,
    pl=False,
    m_name="Model",
    step=0.001,
    return_err=True,
):
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds)
    if (phi_trues is None) & (ph is None):
        raise ValueError("You need to input either the parameter phi_trues or ph.")
    if phi_trues is None:
        phi_trues = phi.phi(trues, ph)
    #trues = trues.values
    tbl = pd.DataFrame(
        {
            "trues": trues,
            "phi_trues": phi_trues,
        }
    )
    tbl = pd.concat([tbl, preds], axis=1)
    ms = list(tbl.columns[2:])
    th = np.arange(0, 1 + step, step)
    errors = []
    for ind in th:
        errors.append(
            [
                sum(
                    tbl.apply(
                        lambda x: ((x["trues"] - x[y]) ** 2)
                        if x["phi_trues"] >= ind
                        else 0,
                        axis=1,
                    )
                )
                for y in ms
            ]
        )

    areas = []
    for x in range(1, len(th)):
        areas.append(
            [step * (errors[x - 1][y] + errors[x][y]) / 2 for y in range(len(ms))]
        )
    areas = pd.DataFrame(data=areas, columns=ms)
    res = areas.apply(lambda x: sum(x))
    names = res.index
    if pl:
        plt.plot(th, errors, label=names)
        plt.ylabel("Error")
        plt.xlabel("Threshold")
        if len(names) > 1:
            plt.legend()
        plt.show()

    # Create a DataFrame with a 'sera' column
    sera_df = pd.DataFrame({"sera": res})

    if return_err:
        return {"sera_df": sera_df, "errors": errors, "thrs": th}
    else:
        return sera_df

# %%
def autophi(y, out_dtype_np=np.float32):
    y = np.float64(np.array(y))
    phi_parms = phi_control(pd.Series(y))
    phi_values = phi(pd.Series(y), phi_parms=phi_parms)
    return torch.Tensor(np.array(phi_values, dtype=out_dtype_np))

# %%
def sert(y, y_hat, phi, t, include_equals=True,device="cpu"):
    #y -> true y's
    #y_hat -> predicted y's
    #phi -> relevances for true y's (compute with phi function)
    #t -> relevance threshold
    #include_equals -> whether to include observations where relevance == threshold

    #Ribeiro, R.P., Moniz, N. Imbalanced regression and extreme value prediction.
    #Mach Learn 109, 1803–1835 (2020). https://doi.org/10.1007/s10994-020-05900-9
    #SER_t = sum((y_hat - y)**2), y_hat, y from D_t, where D_t are values
    #with relevance ??at?? or below threshold t

    #generate zero replacements
    #1's if we include the exact threshold values
    #0's otherwise
    if include_equals:
        zr = torch.ones(y.shape[0])
    else:
        zr = torch.zeros(y.shape[0])

    #compute the value mask:
    #1 - compute differences between t and phi
    mask = phi - t
    #2 - replace values that are less then 0 (relevance below threshold) with 0's
    #values that are more than 0 with 1's
    #exact zeros with the zero replacement values (depending on the parameters)
    if device=="cuda":
        mask = torch.heaviside(mask, zr.cuda()).cuda()
    else:
        mask = torch.heaviside(mask, zr)

    #we can now compute the squared differences for each point in y
    #the irrelevant values are zeroed out and will not contribute
    sq_err = (y_hat - y) ** 2

    #and now remove contribution of irrelevent values
    sq_err = sq_err * mask

    #finally, we can sum the value and return it
    return torch.sum(sq_err)


# %%
def sert_t_vectorizer(y, y_hat, phi, include_equals=True,device="cpu"):
    #uses functorch and closures to create a fast
    #version of the sert function that will evaluate it for
    #different t's
    #returns a function that should be called with a vector of t's
    #y -> true y's
    #y_hat -> predicted y's
    #phi -> relevances for true y's (compute with phi function)
    #include_equals -> whether to include observations where relevance == threshold

    def local_sert(t):
        return sert(y, y_hat, phi, t, include_equals,device)
    return torch.vmap(local_sert, 0, 0)

# %%
def sera_errors_pt(y, y_hat, phi, include_equals=True, resolution=1000):
    #y -> true y's
    #y_hat -> predicted y's
    #phi -> relevances for true y's (compute with phi function)
    #include_equals -> whether to include observations where relevance == threshold
    #resolution -> how much points to use for numerical integration of SER_t

    #Ribeiro, R.P., Moniz, N. Imbalanced regression and extreme value prediction.
    #Mach Learn 109, 1803–1835 (2020). https://doi.org/10.1007/s10994-020-05900-9

    #we will evaluate the function and compute the integral numerically
    #prepare the vectorized function
    sert_v = sert_t_vectorizer(y, y_hat, phi, include_equals)

    #generate a list of points for sert eval
    ts = torch.linspace(0, 1, resolution)

    #eval sert on candidate t's
    sert_points = sert_v(ts)

    return sert_points, ts

def sera_pt(y, y_hat, phi, include_equals=True, resolution=1000,device="cpu"):
    #y -> true y's
    #y_hat -> predicted y's
    #phi -> relevances for true y's (compute with phi function)
    #include_equals -> whether to include observations where relevance == threshold
    #resolution -> how much points to use for numerical integration of SER_t

    #Ribeiro, R.P., Moniz, N. Imbalanced regression and extreme value prediction.
    #Mach Learn 109, 1803–1835 (2020). https://doi.org/10.1007/s10994-020-05900-9
    #SERA = integral[0,1](SER_t(y, y_hat, phi, t))dt

    #we will evaluate the function and compute the integral numerically
    #prepare the vectorized function
    sert_v = sert_t_vectorizer(y, y_hat, phi, include_equals,device)

    #generate a list of points for sert eval
    if device=="cuda":
        ts = torch.linspace(0, 1, resolution).cuda()
        sert_points = sert_v(ts).cuda()
    else:
        ts = torch.linspace(0, 1, resolution)

        #eval sert on candidate t's
        sert_points = sert_v(ts)

    #use trapezoid to numerically compute integral
    sera = torch.trapezoid(sert_points, ts)

    return sera




class LinReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.lr = nn.Linear(2,1)

    def forward(self, x):
        x = self.lr(x)
        return x



class SeraCriterion(nn.Module):
    def __init__(self, ph, resolution = 1000,device="cpu"):
        super().__init__()
        self.ph=ph
        self._res = resolution
        self.device=device

    def forward(self, y_hat,y):
        y=y.cpu()
        phi_values=phi(pd.Series(y.flatten(),dtype="float64"), phi_parms=self.ph)
        phi_values=torch.Tensor(np.array(phi_values, dtype=np.float32))
        if self.device=="cuda":
            y=y.cuda()
            y_hat=y_hat.cuda()
            phi_values=phi_values.cuda()
        return sera_pt(y, y_hat, phi_values, resolution=self._res,device=self.device)



