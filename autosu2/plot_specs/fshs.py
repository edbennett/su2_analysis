from numpy import asarray, linspace, nan, ravel
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from uncertainties import ufloat

from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, format_value_and_error
from ..derived_observables import merge_no_w0

from .common import beta_colour_marker, add_figure_key, preliminary


def sm_residual(gamma_s, data, count_valid_points=False):
    beta_data = data.copy()
    beta_data["fshs_x"] = beta_data.L ** (1 + gamma_s) * beta_data.value_mpcac_mass
    beta_data["LM_H"] = beta_data.L * beta_data.value_g5_mass

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

    if count_valid_points:
        return P_b / valid_point_count, valid_point_count
    else:
        return P_b / valid_point_count


def do_plot(betas, fit_results, merged_data, Nf):
    filename = f"final_plots/fshs_Nf{Nf}.pdf"
    fig, axes = plt.subplots(
        ncols=len(betas),
        figsize=(2.5 + 1.5 * len(betas), 3.5),
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    for fit_beta, ax in zip(betas, axes):
        gamma = fit_results[fit_beta]
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

        ax.set_title(f"$\\beta={fit_beta} \\Rightarrow \\gamma_*={gamma:.3}$")

    add_figure_key(fig, Nf=Nf)

    for ax in axes:
        ax.set_xlabel(r"$L (am_{\mathrm{PCAC}})^{1/(1+\gamma_*)}$")
        ax.set_xlim((0, None))

    axes[0].set_ylabel("$L aM_{2^+_{\mathrm{s}}}$")
    axes[0].set_ylim((0, None))

    fig.tight_layout(pad=0.08, h_pad=0.5, rect=(0, 0, 1, 0.92))

    fig.savefig(filename)
    plt.close(fig)


def do_table(results, merged_data, Nf):
    filename = f"fshs_gamma_Nf{Nf}.tex"
    columns = r"$\beta$", None, r"$\gamma_*$", r"$N_{\mathrm{points}}$"
    table_content = []

    for beta, result in sorted(results.items()):
        gamma_s = result["x"][0]
        gamma_s_err = result["hess_inv"][0, 0]
        _, valid_point_count = sm_residual(
            gamma_s, merged_data[merged_data.beta == beta], count_valid_points=True
        )
        formatted_gamma_s = format_value_and_error(gamma_s, gamma_s_err)
        table_content.append(f"    {beta} & {formatted_gamma_s} & {valid_point_count}")
    generate_table_from_content(filename, table_content, columns)


def generate_single_Nf(data, Nf, betas_to_plot):
    data = data[data.Nf == Nf]

    observables = "g5_mass", "gk_mass"
    extra_observables = ("mpcac_mass", "g5_decay_const")
    observable_labels = r"\gamma_5", r"\gamma_k"

    merged_data = merge_no_w0(data, observables + extra_observables).dropna(
        subset=("value_mpcac_mass", "value_g5_mass")
    )

    fit_results = {}
    for beta, _, _ in beta_colour_marker[Nf]:
        fit_results[beta] = minimize(
            sm_residual, 1.0, args=merged_data[merged_data.beta == beta]
        )

    do_plot(
        betas_to_plot,
        {beta: fit_results[beta]["x"][0] for beta, _, _ in beta_colour_marker[Nf]},
        merged_data,
        Nf,
    )
    do_table(fit_results, merged_data, Nf)


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.3, preliminary=preliminary)
    generate_single_Nf(data, 1, [2.05, 2.2, 2.4])
    generate_single_Nf(data, 2, [2.35])
