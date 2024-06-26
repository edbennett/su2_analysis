#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lsqfit
import gvar as gv

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import add_figure_key, beta_colour_marker, preliminary
from .w0_chiral import fit_1_over_w0


def fit_form_linear(x, p):
    return x * p["m"] + p["q"]


def fit_form_quadratic(x, p):
    return p["q"] + p["c0"] * x + p["c1"] * x**2


def fit_form_invexp(x, p):
    # return p["d1"] * gv.exp(p["d2"] * (1 / (x - p["b0"])**2)) + p["d0"]
    return p["d1"] * gv.exp(p["d2"] * (1 / (1 / x - p["beta0"])) ** 2) + p["d0"]


def get_fit_form_priors(x_var):
    if x_var == "value_w0":
        fit_form = fit_form_linear
        fit_formula = r"$\gamma_*(w_0) = q + \frac{m}{w_0}$"
        mapping = {"q": "q", "m": "m"}
        prior = {"m": gv.gvar(0.5, 0.5), "q": gv.gvar(0.5, 0.5)}
    elif x_var == "value_chiral_w0":
        fit_form = fit_form_quadratic
        fit_formula = r"$\gamma_*(w_0^\chi) = \gamma_*^{\mathrm{cont.}} + c_0\frac{a}{w_0} + c_1\frac{a^2}{w_0^2}$"
        mapping = {"q": "q", "c0": "c_0", "c1": "c_1"}
        prior = {
            "q": gv.gvar(0.5, 0.5),
            "c0": gv.gvar(2.5, 2.5),
            "c1": gv.gvar(0.0, 5.0),
        }
    elif x_var == "beta":
        fit_form = fit_form_invexp
        fit_formula = r"$\gamma_*(\beta) = d_0 + d_1 \exp\left( \frac{d_2}{\left(\beta - \beta_0\right)^2}\right)$"
        mapping = {"d0": "d_0", "d1": "d_1", "d2": "d_2", "beta0": r"\beta_0"}
        prior = {
            "d0": gv.gvar(0.5, 0.5),
            "d1": gv.gvar(0.5, 0.5),
            "d2": gv.gvar(0.0, 0.5),
            "beta0": gv.gvar(2.0, 2.0),
        }
    else:
        raise ValueError(f"Can't fit {x_var}")

    return fit_form, fit_formula, mapping, prior


def fit(data, x_var="value_w0"):
    x_data = 1 / data[x_var].values

    gamma_error_combined = (
        data.uncertainty_gamma_aic**2 + data.value_gamma_aic_syst**2
    ) ** 0.5
    y_data = gv.gvar(data.value_gamma_aic.values, gamma_error_combined.values)

    fit_form, _, _, prior = get_fit_form_priors(x_var)
    return lsqfit.nonlinear_fit(
        data=(x_data, y_data),
        fcn=fit_form,
        prior=prior,
    )


def plot(
    data,
    fit_result,
    Nf,
    xlabel_slug="a / w_0",
    x_var="value_w0",
    xerr_var=None,
    add_label=False,
):
    filename = f"assets/plots/continuum_gammastar_Nf{Nf}_{x_var}.pdf"

    fig, ax = plt.subplots(layout="constrained", figsize=(3.5, 3))

    for beta, colour, marker in beta_colour_marker[Nf]:
        subset = data[data.beta == beta]
        ax.errorbar(
            1 / subset[x_var],
            subset.value_gamma_aic,
            xerr=subset[xerr_var] if xerr_var is not None else None,
            yerr=subset.uncertainty_gamma_aic,
            marker=marker,
            ls="none",
            color=colour,
        )

    fit_form, fit_formula, mapping, _ = get_fit_form_priors(x_var)
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

    ax.set_xlabel(f"${xlabel_slug}$")
    ax.set_ylabel(r"$\gamma_*$")

    if add_label:
        fig.suptitle(fit_formula)
        ax.set_title(
            ", ".join(
                f"${label} = {fit_result.p[name]}$" for name, label in mapping.items()
            ),
            fontsize="x-small",
        )

    add_figure_key(fig, Nf=Nf, nrow=2)

    fig.savefig(filename)


def generate_single_Nf(
    data, Nf, xlabel_slug="a / w_0", x_var="w0", xerr_var=None, exclude=None
):
    if exclude is None:
        exclude = []

    hatted_data = merge_and_hat_quantities(
        data,
        (
            "gamma_aic",
            "gamma_aic_syst",
        ),
    ).dropna(subset=["value_gamma_aic"])
    subset_data = hatted_data[hatted_data.Nf == Nf]
    filtered_data = subset_data[~subset_data.label.isin(exclude)]
    fit_result = fit(filtered_data, x_var=x_var)
    print(fit_result)

    plot(
        subset_data,
        fit_result,
        Nf,
        x_var=x_var,
        xerr_var=xerr_var,
        xlabel_slug=xlabel_slug,
    )


def add_w0_extrapolation(data):
    modified_data = data.copy()
    modified_data["value_chiral_w0"] = np.nan
    modified_data["uncertainty_chiral_w0"] = np.nan

    for Nf in set(data.Nf):
        Nf_subset = data[data.Nf == Nf]
        for beta in set(Nf_subset.beta):
            popt, pcov = fit_1_over_w0(data, Nf, beta)
            index = (data.Nf == Nf) & (data.beta == beta)
            modified_data.loc[index, "value_chiral_w0"] = 1 / popt[0]
            modified_data.loc[index, "uncertainty_chiral_w0"] = (
                pcov[0, 0] ** 0.5 * popt[0] ** 2
            )

    return modified_data


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=1, linewidth=0.5, preliminary=preliminary)

    generate_single_Nf(
        data,
        Nf=1,
        x_var="value_w0",
        xerr_var="uncertainty_w0",
        exclude=["DB4M13", "DB7M7"],
    )
    generate_single_Nf(
        data, Nf=1, x_var="beta", exclude=["DB4M13", "DB7M7"], xlabel_slug=r"1 / \beta"
    )

    data_with_w0_extrapolation = add_w0_extrapolation(data)
    generate_single_Nf(
        data_with_w0_extrapolation,
        Nf=1,
        x_var="value_chiral_w0",
        xerr_var="uncertainty_chiral_w0",
        xlabel_slug=r"a / w_0^\chi",
    )
