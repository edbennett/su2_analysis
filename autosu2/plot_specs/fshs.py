import csv

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from format_multiple_errors import format_multiple_errors
from uncertainties import ufloat

from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, format_value_and_error
from ..derived_observables import merge_no_w0
from ..provenance import text_metadata, get_basic_metadata, number_to_latex

from .common import beta_colour_marker, add_figure_key, preliminary


csv_filename = "processed_data/gammastar_fshs.csv"
definition_filename = "assets/definitions/gammastar_fshs.tex"


def sm_residual(gamma_s, data, count_valid_points=False, observable="value_g5_mass"):
    beta_data = data.copy()
    beta_data["fshs_x"] = beta_data.L ** (1 + gamma_s) * beta_data.value_mpcac_mass
    beta_data["LM_H"] = beta_data.L * beta_data[observable]

    valid_point_count = 0
    P_b = 0

    for L in set(beta_data.L):
        set_p = beta_data[beta_data.L == L]
        set_j = beta_data[beta_data.L != L]

        for _, point_i in set_j.iterrows():
            if (point_i.fshs_x < min(set_p.fshs_x)) or (
                point_i.fshs_x > max(set_p.fshs_x)
            ):
                continue

            point_below = set_p[
                set_p.fshs_x == set_p[set_p.fshs_x < point_i.fshs_x].fshs_x.max()
            ]
            point_above = set_p[
                set_p.fshs_x == set_p[set_p.fshs_x > point_i.fshs_x].fshs_x.min()
            ]

            LM_H_below = float(point_below.LM_H.iloc[0])
            LM_H_above = float(point_above.LM_H.iloc[0])
            x_below = float(point_below.fshs_x.iloc[0])
            x_above = float(point_above.fshs_x.iloc[0])

            interp_LM_H = LM_H_below + (point_i.fshs_x - x_below) * (
                LM_H_above - LM_H_below
            ) / (x_above - x_below)

            P_b += (point_i.LM_H - interp_LM_H) ** 2
            valid_point_count += 1

    if valid_point_count == 0:
        result = np.inf
    else:
        result = P_b / valid_point_count

    if count_valid_points:
        return result, valid_point_count
    else:
        return result


def result_to_ufloat(result):
    return ufloat(result["x"][0], result["hess_inv"][0, 0])


def do_plot(betas, fit_results, merged_data, Nf):
    filename = f"assets/plots/fshs_Nf{Nf}.pdf"
    fig, axes = plt.subplots(
        ncols=len(betas),
        figsize=(2.5 + 1.5 * len(betas), 3.5),
        sharey=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes.ravel()

    for fit_beta, ax in zip(betas, axes):
        gamma_with_error = result_to_ufloat(fit_results[fit_beta])
        gamma = gamma_with_error.nominal_value
        for beta, colour, marker in beta_colour_marker[Nf]:
            subset_data = merged_data[merged_data.beta == beta]
            L = subset_data.L
            ax.errorbar(
                L * subset_data.value_mpcac_mass ** (1 / (1 + gamma)),
                L * subset_data.value_g5_mass,
                xerr=(
                    L
                    * subset_data.value_mpcac_mass ** (-gamma / (1 + gamma))
                    * subset_data.uncertainty_mpcac_mass
                    / (1 + gamma)
                ),
                yerr=L * subset_data.uncertainty_g5_mass,
                marker=marker,
                color=colour,
                ls="none",
            )

        ax.set_title(
            f"$\\beta={fit_beta} \\Rightarrow \\gamma_*={gamma_with_error:.2uSL}$",
        )

    add_figure_key(fig, Nfs=[Nf])

    for ax in axes:
        ax.set_xlabel(r"$L (am_{\mathrm{PCAC}})^{1/(1+\gamma_*)}$")
        ax.set_xlim((0, None))

    axes[0].set_ylabel(r"$L aM_{2^+_{\mathrm{s}}}$")
    axes[0].set_ylim((0, None))

    fig.savefig(filename)
    plt.close(fig)


def do_table(results, merged_data, Nf, ensembles):
    filename = f"gamma_Nf{Nf}.tex"
    columns = (
        r"$\beta$",
        None,
        r"$\gamma_*$ (FSHS)",
        r"$N_{\mathrm{points}}$",
        None,
        r"$\gamma_*$ (AIC)",
        "Ensemble",
    )
    table_content = []

    for beta, result in sorted(results.items()):
        beta_data = merged_data[(merged_data.beta == beta) & (merged_data.Nf == Nf)]
        gamma_s = result["x"][0]
        gamma_s_err = result["hess_inv"][0, 0]
        _, valid_point_count = sm_residual(gamma_s, beta_data, count_valid_points=True)
        formatted_gamma_s = format_value_and_error(gamma_s, gamma_s_err)

        gamma_s_aics = beta_data.dropna(
            subset=["value_gamma_aic", "uncertainty_gamma_aic", "value_gamma_aic_syst"]
        )
        num_rows = len(gamma_s_aics)

        if num_rows == 0:
            table_content.append(
                f"{beta} & {formatted_gamma_s} & {valid_point_count} & $\\cdots$ & $\\cdots$"
            )
            continue

        row_starts = [
            f"\\multirow{{{num_rows}}}{{*}}{{{beta}}} & "
            f"\\multirow{{{num_rows}}}{{*}}{{{formatted_gamma_s}}} & "
            f"\\multirow{{{num_rows}}}{{*}}{{{valid_point_count}}}"
        ] + [" & &"] * (num_rows - 1)
        for row_start, (_, row) in zip(
            row_starts, gamma_s_aics.sort_values(by="m", ascending=False).iterrows()
        ):
            formatted_gamma_s_aic = format_multiple_errors(
                row.value_gamma_aic,
                row.uncertainty_gamma_aic,
                row.value_gamma_aic_syst,
                abbreviate=True,
                latex=True,
            )
            table_content.append(f"{row_start} & {formatted_gamma_s_aic} & {row.label}")

    preamble = text_metadata(
        get_basic_metadata(ensembles["_filename"]), comment_char="%"
    )
    generate_table_from_content(filename, table_content, columns, preamble=preamble)


def single_fit(merged_data, Nf, beta):
    return minimize(
        sm_residual,
        1.0,
        args=merged_data[(merged_data.beta == beta) & (merged_data.Nf == Nf)],
    )


def gammastar_fshs(merged_data, Nf, beta):
    result = single_fit(
        merged_data.dropna(subset=["value_mpcac_mass", "value_g5_mass"]), Nf, beta
    )
    return result_to_ufloat(result)


def write_definitions(fit_results, Nf):
    for beta, result in fit_results.items():
        latex_var_name = (
            f"GammaStarFSHSNf{number_to_latex(Nf)}Beta{number_to_latex(beta)}"
        )
        gammastar = result_to_ufloat(result)
        with open(definition_filename, "a") as f:
            print(f"\\newcommand \\{latex_var_name} {{{gammastar:.02uSL}}}", file=f)


def write_csv(fit_results, Nf):
    with open(csv_filename, "a") as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        for beta, result in fit_results.items():
            csv_writer.writerow(
                (
                    Nf,
                    beta,
                    result["x"][0],
                    result["hess_inv"][0, 0],
                )
            )


def generate_single_Nf(data, Nf, betas_to_plot, ensembles):
    data = data[data.Nf == Nf]

    observables = "g5_mass", "gk_mass"
    extra_observables = ("mpcac_mass", "g5_decay_const", "gamma_aic", "gamma_aic_syst")

    merged_data = merge_no_w0(data, observables + extra_observables).dropna(
        subset=("value_mpcac_mass", "value_g5_mass")
    )

    fit_results = {}
    for beta, _, _ in beta_colour_marker[Nf]:
        fit_results[beta] = single_fit(merged_data, Nf, beta)

    do_plot(
        betas_to_plot,
        {beta: fit_results[beta] for beta, _, _ in beta_colour_marker[Nf]},
        merged_data,
        Nf,
    )
    do_table(fit_results, merged_data, Nf, ensembles)
    write_definitions(fit_results, Nf)
    write_csv(fit_results, Nf)


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.3, preliminary=preliminary)
    ensembles_metadata = get_basic_metadata(ensembles["_filename"])
    with open(definition_filename, "w") as f:
        print(text_metadata(ensembles_metadata, comment_char="%"), file=f)

    with open(csv_filename, "w") as f:
        print(text_metadata(ensembles_metadata, comment_char="#"), file=f)
        print("Nf,beta,gamma_value,gamma_uncertainty", file=f)

    generate_single_Nf(data, 1, [2.05, 2.2, 2.4], ensembles)
    generate_single_Nf(data, 2, [2.25, 2.3, 2.35], ensembles)
