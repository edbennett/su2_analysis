#!/usr/bin/env python

import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import beta_colour_marker, preliminary


def plot_single_ratio(data, channel, label, filename):
    fig, ax = plt.subplots(figsize=(3.5, 3.5), layout="constrained")

    ax.set_xlabel(r"$w_0 m_{\mathrm{PCAC}}$")
    ax.set_ylabel(f"$\\frac{{M_{{{label}}}}}{{M_{{2^+_{{\mathrm{{s}}}}}}}}$")

    for Nf in 1, 2:
        for beta, colour, marker in beta_colour_marker[Nf]:
            data_to_plot = data[
                (data.beta == beta) & ~(data.label.str.endswith("*")) & (data.Nf == Nf)
            ]
            if data_to_plot[f"value_{channel}_ratio"].isnull().all():
                continue

            ax.errorbar(
                data_to_plot.value_mpcac_mass_hat,
                data_to_plot[f"value_{channel}_ratio"],
                xerr=data_to_plot.uncertainty_mpcac_mass_hat,
                yerr=data_to_plot[f"uncertainty_{channel}_ratio"],
                color=colour,
                marker=marker,
                label=f"$N_{{\\mathrm{{f}}}}={Nf}, \\beta={beta}$",
                ls="none",
            )

    fig.legend(
        ncol=2,
        columnspacing=1,
        handletextpad=0,
        borderpad=0,
        loc="outside upper center",
        frameon=False,
    )

    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    fig.savefig(filename)
    plt.close(fig)


def generate(data, ensembles):
    set_plot_defaults(markersize=2, capsize=1, linewidth=0.5, preliminary=preliminary)

    hatted_data = merge_and_hat_quantities(
        data, ("A1++_mass", "spin12_mass", "g5_mass", "mpcac_mass")
    )
    for channel, label, filename in [
        ("A1++", r"0^{++}", "final_plots/scalar_ratio.pdf"),
        ("spin12", r"\breve{g}", "final_plots/spin12_ratio.pdf"),
    ]:
        hatted_data[f"value_{channel}_ratio"] = (
            hatted_data[f"value_{channel}_mass"] / hatted_data.value_g5_mass
        )
        hatted_data[f"uncertainty_{channel}_ratio"] = (
            hatted_data[f"uncertainty_{channel}_mass"] ** 2
            / hatted_data.value_g5_mass**2
            + hatted_data[f"value_{channel}_mass"] ** 2
            * hatted_data.uncertainty_g5_mass**2
            / hatted_data.value_g5_mass**2
        ) ** 0.5

        plot_single_ratio(hatted_data, channel, label, filename)
