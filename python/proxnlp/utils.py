"""
Copyright (C) 2022 LAAS-CNRS, INRIA
"""
import numpy as np

import matplotlib.pyplot as plt


_ROOT_10 = 10.0**0.5


def plot_pd_errs(ax0: plt.Axes, prim_errs, dual_errs):
    prim_errs = np.asarray(prim_errs)
    dual_errs = np.asarray(dual_errs)
    ax0.plot(prim_errs, c="tab:blue")
    ax0.set_xlabel("Iterations")
    col2 = "tab:orange"
    ax0.plot(dual_errs, c=col2)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_color(col2)
    ax0.yaxis.label.set_color(col2)
    ax0.set_yscale("log")
    ax0.legend(["Primal error $p$", "Dual error $d$"])
    ax0.set_title("Solver primal-dual residuals")

    # handle scaling
    yhigh = ax0.get_ylim()[1]
    if len(prim_errs) == 0 or len(dual_errs) == 0:
        return
    mach_eps = np.finfo(float).eps
    dmask = dual_errs > 2 * mach_eps
    pmask = prim_errs > 2 * mach_eps
    ymin = np.finfo(float).max
    if dmask.any():
        ymin = np.min(dual_errs[dmask])
    if pmask.any() and sum(prim_errs > 0) > 0:
        ymin = min(np.min(prim_errs[pmask]), ymin)
    ax0.set_ylim(ymin / _ROOT_10, yhigh)
