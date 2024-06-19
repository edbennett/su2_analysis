#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lsqfit
import gvar as gv

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import add_figure_key, beta_colour_marker, preliminary


def fit_form(x, p):
    return x * p["m"] + p["q"]


def fit(data):
    x_data = 1 / data.value_w0

    gamma_error_combined = (
        data.uncertainty_gamma_aic**2 + data.value_gamma_aic_syst**2
    ) ** 0.5
    y_data = gv.gvar(data.value_gamma_aic.values, gamma_error_combined.values)
    return lsqfit.nonlinear_fit(
        data=(x_data, y_data),
        fcn=fit_form,
        prior={"m": gv.gvar(0.5, 0.5), "q": gv.gvar(0.5, 0.5)},
    )


def plot(data, fit_result, Nf):
    filename = f"final_plots/continuum_gammastar_Nf{Nf}.pdf"

    fig, ax = plt.subplots(layout="constrained", figsize=(3.5, 3))

    for beta, colour, marker in beta_colour_marker[Nf]:
        subset = data[data.beta == beta]
        ax.errorbar(
            1 / subset.value_w0,
            subset.value_gamma_aic,
            yerr=subset.uncertainty_gamma_aic,
            marker=marker,
            ls="none",
            color=colour,
        )

    _, xmax = ax.get_xlim()
    x_range = np.linspace(0, xmax, 1000)
    y_values = fit_form(x_range, fit_result.p)
    ax.plot(x_range, gv.mean(y_values))
    ax.fill_between(
        x_range,
        gv.mean(y_values) - gv.sdev(y_values),
        gv.mean(y_values) + gv.sdev(y_values),
        color="gray",
        alpha=0.1,
    )
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)

    ax.set_xlabel(r"$a / w_0$")
    ax.set_ylabel(r"$\gamma^*$")

    add_figure_key(fig, Nf=Nf, nrow=2)

    fig.savefig(filename)


def generate_single_Nf(data, Nf, exclude=None):
    if exclude is None:
        exclude = []

    set_plot_defaults(markersize=3, capsize=1, linewidth=0.5, preliminary=preliminary)

    hatted_data = merge_and_hat_quantities(
        data,
        (
            "gamma_aic",
            "gamma_aic_syst",
        ),
    ).dropna(subset=["value_gamma_aic"])
    subset_data = hatted_data[hatted_data.Nf == Nf]
    filtered_data = subset_data[~subset_data.label.isin(exclude)]
    fit_result = fit(filtered_data)
    plot(subset_data, fit_result, Nf)


def generate(data, ensembles):
    generate_single_Nf(data, Nf=1, exclude=["DB4M13", "DB7M7"])
