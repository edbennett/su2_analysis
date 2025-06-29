import matplotlib.pyplot as plt
from pandas import read_csv

from ..derived_observables import merge_quantities
from ..plots import set_plot_defaults

from .common import add_figure_key, beta_colour_marker, preliminary, TWO_COLUMN


def uncertainty_ratio_sqrtsigma(suscept, suscept_err, sqrtsigma, sqrtsigma_err):
    return (
        (0.25 * suscept**-0.75 * suscept_err / sqrtsigma) ** 2
        + (suscept**0.25 * sqrtsigma_err / sqrtsigma**2) ** 2
    ) ** 0.5


def add_sideload_data(ax):
    """
    Add data quoted from literature to the ax; specifically from
    doi:10.1140/epjc/s10052-013-2426-6 (arXiv:1209.5579)
    """

    data = read_csv("external_data/su2_topology.csv", comment="#")

    for Nf, label_suffix, marker in [(0, "", "s")]:
        subset_data = data[data.Nf == Nf]
        ax.errorbar(
            subset_data.sqrtsigma**2,
            subset_data.suscept**0.25 / subset_data.sqrtsigma,
            xerr=2 * subset_data.sqrtsigma * subset_data.sqrtsigma_err,
            yerr=uncertainty_ratio_sqrtsigma(
                subset_data.suscept,
                subset_data.suscept_err,
                subset_data.sqrtsigma,
                subset_data.sqrtsigma_err,
            ),
            label=f"$N_{{\\mathrm{{f}}}} = {Nf}{label_suffix}$ (1209.5579)",
            ls="none",
            marker=marker,
        )


def generate(data, ensembles):
    set_plot_defaults(markersize=2, capsize=0.2, linewidth=0.5, preliminary=preliminary)

    filename = "assets/plots/susceptibility.pdf"
    fig, axes = plt.subplots(
        figsize=(TWO_COLUMN, 5), ncols=2, sharey=True, sharex=True, layout="constrained"
    )
    merged_data = merge_quantities(
        data, ("sqrtsigma", "chi_top", "fitted_Q0", "Q_width", "Q_tau_exp")
    )
    with_sqrtsigma = True

    merged_data["value_chi_top_14_hat"] = (
        merged_data.value_chi_top**0.25 * merged_data.value_w0
    )
    merged_data["uncertainty_chi_top_14_hat"] = (
        (
            0.25
            * merged_data.value_chi_top**-0.75
            * merged_data.uncertainty_chi_top
            * merged_data.value_w0
        )
        ** 2
        + (merged_data.value_chi_top**0.25 * merged_data.uncertainty_w0) ** 2
    ) ** 0.5

    merged_data["value_chi_top_14_over_sqrtsigma"] = (
        merged_data.value_chi_top**0.25 / merged_data.value_sqrtsigma
    )
    merged_data["uncertainty_chi_top_14_over_sqrtsigma"] = uncertainty_ratio_sqrtsigma(
        merged_data.value_chi_top,
        merged_data.uncertainty_chi_top,
        merged_data.value_sqrtsigma,
        merged_data.uncertainty_sqrtsigma,
    )

    if with_sqrtsigma:
        axes[0].set_ylabel(r"$\chi^{\frac{1}{4}} / \sqrt{\sigma}$")
    else:
        axes[0].set_ylabel(r"$w_0 \chi^{\frac{1}{4}}$")

    for Nf, ax in zip([1, 2], axes):
        ax.set_xlabel(r"$a^2 \sigma$")
        ax.set_title(f"$N_{{\\mathrm{{f}}}} = {Nf}$")
        for beta, colour, marker in beta_colour_marker[Nf]:
            data_to_plot = merged_data[
                (merged_data.beta == beta)
                & ~(merged_data.label.str.endswith("*"))
                & (merged_data.Nf == Nf)
            ]
            x_values = data_to_plot.value_sqrtsigma**2
            x_uncertainties = (
                2 * data_to_plot.value_sqrtsigma * data_to_plot.uncertainty_sqrtsigma
            )
            if with_sqrtsigma:
                y_values = data_to_plot.value_chi_top_14_over_sqrtsigma
                y_uncertainties = data_to_plot.uncertainty_chi_top_14_over_sqrtsigma
            else:
                y_values = data_to_plot.value_chi_top_14_hat
                y_uncertainties = data_to_plot.uncertainty_chi_top_14_hat

            ax.errorbar(
                x_values,
                y_values,
                xerr=x_uncertainties,
                yerr=y_uncertainties,
                color=colour,
                marker=marker,
                ls="none",
            )

        if with_sqrtsigma:
            add_sideload_data(ax)

    add_figure_key(fig, Nfs=[1, 2], shortlabel=True)
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    fig.savefig(filename)
    plt.close(fig)
