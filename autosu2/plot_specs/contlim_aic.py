#!/usr/bin/env python

from dataclasses import dataclass
from functools import partial
import typing

import numpy as np
import matplotlib.pyplot as plt
import lsqfit
import gvar as gv
from uncertainties import ufloat

from ..fit_glue import weighted_mean
from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities
from ..provenance import latex_metadata, get_basic_metadata, number_to_latex

from .common import add_figure_key, beta_colour_marker, preliminary
from .w0_chiral import fit_1_over_w0


# Some fit forms break at the zero point exactly,
# so use an infinitesimal value instead
ALMOST_ZERO = 1e-31

definition_filename = "assets/definitions/gammastar.tex"


@dataclass
class FitForm:
    fit_function: typing.Callable
    fit_formula: str
    mapping: dict[str, str]
    prior: dict[str, gv.gvar]
    latex_name: str
    legend_label: str = ""

    def __call__(self, *args, **kwargs):
        return self.fit_function(*args, **kwargs)


def fit_form_linear(x, p):
    return x * p["m"] + p["q"]


def fit_form_quadratic(x, p):
    return p["q"] + p["c0"] * x + p["c1"] * x**2


def fit_form_invexp(x, p):
    return p["d1"] * gv.exp(-p["d2"] * (1 / x)) + p["d0"]


fit_forms = {
    "value_w0": [
        FitForm(
            fit_function=fit_form_linear,
            fit_formula=r"$\gamma_*(w_0) = q + \frac{m}{w_0}$",
            mapping={"q": "q", "m": "m"},
            prior={"m": gv.gvar(0.5, 0.5), "q": gv.gvar(0.5, 0.5)},
            latex_name="WZero",
        ),
    ],
    "value_chiral_w0": [
        FitForm(
            fit_function=fit_form_quadratic,
            fit_formula=r"$\gamma_*(w_0^\chi) = \gamma_*^{\mathrm{cont.}} + c_0\frac{a}{w_0} + c_1\frac{a^2}{w_0^2}$",
            mapping={"q": "q", "c0": "c_0", "c1": "c_1"},
            prior={
                "q": gv.gvar(0.5, 0.5),
                "c0": gv.gvar(2.5, 2.5),
                "c1": gv.gvar(0.0, 5.0),
            },
            latex_name="WZeroChiral",
        ),
    ],
    "beta": [
        FitForm(
            fit_function=partial(fit_form_invexp),
            fit_formula=r"$\gamma_*(\beta) = d_0 + d_1 \exp\left(-d_2 \beta\right)$",
            mapping={"d0": "d_0", "d1": "d_1", "d2": "d_2"},
            prior={
                "d0": gv.gvar(0.5, 0.5),
                "d1": gv.gvar(0.0, 200.0),
                "d2": gv.gvar(0.0, 0.5),
            },
            latex_name="BetaLinearExponent",
        ),
    ],
}


def fit(data, x_var="value_w0"):
    x_data = 1 / data[x_var].values
    target_fit_forms = fit_forms[x_var]

    gamma_error_combined = (
        data.uncertainty_gamma_aic**2 + data.value_gamma_aic_syst**2
    ) ** 0.5
    y_data = gv.gvar(data.value_gamma_aic.values, gamma_error_combined.values)

    return [
        lsqfit.nonlinear_fit(
            data=(x_data, y_data),
            fcn=fit_form,
            prior=fit_form.prior,
        )
        for fit_form in target_fit_forms
    ]


def plot(
    data,
    fit_results,
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

    target_fit_forms = fit_forms[x_var]
    _, xmax = ax.get_xlim()
    x_range = np.linspace(0, xmax, 1000)

    fit_colours = "C0", "C1", "C2"
    fit_dashes = (1, 1), (3, 1), (None, None)
    fit_handles = []
    for fit_form, fit_result, colour, dash in zip(
        target_fit_forms, fit_results, fit_colours, fit_dashes
    ):
        y_values = fit_form(x_range, fit_result.p)
        fit_handles.append(
            ax.plot(
                x_range,
                gv.mean(y_values),
                label=fit_form.legend_label.format(**fit_result.p),
                color=colour,
                dashes=dash,
            )[0]
        )
        band_lower_bound = gv.mean(y_values) - gv.sdev(y_values)
        band_upper_bound = gv.mean(y_values) + gv.sdev(y_values)

        ax.plot(x_range, band_lower_bound, color=colour, dashes=dash, alpha=0.3)
        ax.plot(x_range, band_upper_bound, color=colour, dashes=dash, alpha=0.3)
        ax.fill_between(
            x_range,
            band_lower_bound,
            band_upper_bound,
            color=colour,
            alpha=0.1,
        )

    if len(fit_results) > 1:
        ax.legend(handles=fit_handles, loc="best")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, None)

    ax.set_xlabel(f"${xlabel_slug}$")
    ax.set_ylabel(r"$\gamma_*$")

    if add_label:
        fig.suptitle(fit_form.fit_formula)
        ax.set_title(
            ", ".join(
                f"${label} = {fit_result.p[name]}$"
                for name, label in fit_form.mapping.items()
            ),
            fontsize="x-small",
        )

    add_figure_key(fig, Nf=Nf, nrow=2)

    fig.savefig(filename)


def print_definitions(fit_results, Nf, x_var="w0"):
    target_fit_forms = fit_forms[x_var]

    for fit_form, fit_result in zip(target_fit_forms, fit_results):
        latex_var_name = (
            f"GammaStarContinuum{fit_form.latex_name}Nf{number_to_latex(Nf)}"
        )
        latex_var_name_chisquare = f"{latex_var_name}ExtrapolationChisquare"
        gammastar_gv = fit_form(np.asarray([ALMOST_ZERO]), fit_result.p)[0]
        gammastar = ufloat(gammastar_gv.mean, gammastar_gv.sdev)
        chisquare = fit_result.chi2 / fit_result.dof
        with open(definition_filename, "a") as f:
            print(f"\\newcommand \\{latex_var_name} {{{gammastar:.02uSL}}}", file=f)
            print(
                f"\\newcommand \\{latex_var_name_chisquare} {{{chisquare:.02f}}}",
                file=f,
            )


def write_mean_result(fit_results, Nf=1):
    latex_var_name = f"GammaStarContinuumMeanNf{number_to_latex(Nf)}"
    target_fit_forms = fit_forms["beta"][0], fit_forms["value_chiral_w0"][0]
    continuum_values = [
        fit_form(np.asarray([ALMOST_ZERO]), fit_result[0].p)[0]
        for fit_form, fit_result in zip(target_fit_forms, fit_results)
    ]
    mean_gammastar_gv = weighted_mean(continuum_values, error_attr="sdev")

    mean_gammastar = ufloat(mean_gammastar_gv.mean, mean_gammastar_gv.sdev)
    with open(definition_filename, "a") as f:
        print(f"\\newcommand \\{latex_var_name} {{{mean_gammastar:.01uSL}}}", file=f)


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
    fit_results = fit(filtered_data, x_var=x_var)

    plot(
        subset_data,
        fit_results,
        Nf,
        x_var=x_var,
        xerr_var=xerr_var,
        xlabel_slug=xlabel_slug,
        add_label=False,
    )
    print_definitions(fit_results, Nf, x_var=x_var)
    return fit_results


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
    with open(definition_filename, "w") as f:
        print(latex_metadata(get_basic_metadata(ensembles["_filename"])), file=f)

    generate_single_Nf(
        data,
        Nf=1,
        x_var="value_w0",
        xerr_var="uncertainty_w0",
        exclude=["DB4M13", "DB7M7"],
    )
    beta_results = generate_single_Nf(
        data, Nf=1, x_var="beta", exclude=["DB4M13", "DB7M7"], xlabel_slug=r"1 / \beta"
    )

    data_with_w0_extrapolation = add_w0_extrapolation(data)
    chiral_w0_results = generate_single_Nf(
        data_with_w0_extrapolation,
        Nf=1,
        x_var="value_chiral_w0",
        xerr_var="uncertainty_chiral_w0",
        xlabel_slug=r"a / w_0^\chi",
    )

    write_mean_result([beta_results, chiral_w0_results], Nf=1)
