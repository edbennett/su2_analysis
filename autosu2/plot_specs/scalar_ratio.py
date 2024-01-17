#!/usr/bin/env python

import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import beta_colour_marker, preliminary


def generate(data, ensembles):
    set_plot_defaults(markersize=2, capsize=1, linewidth=0.5, preliminary=preliminary)

    filename = "final_plots/scalar_ratio_Nf{Nf}.pdf"
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    hatted_data = merge_and_hat_quantities(data, ("A1++_mass", "g5_mass", "mpcac_mass"))
    hatted_data["value_ratio"] = hatted_data["value_A1++_mass"] / hatted_data.value_g5_mass
    hatted_data["uncertainty_ratio"] = (
        hatted_data["uncertainty_A1++_mass"]**2 / hatted_data.value_g5_mass**2
        + hatted_data["value_A1++_mass"]**2
        * hatted_data.uncertainty_g5_mass**2
        / hatted_data.value_g5_mass**2
    ) ** 0.5
    ax.set_xlabel(r"$w_0 m_{\mathrm{PCAC}}$")
    ax.set_ylabel(r"$\frac{M_{0^{++}}}{M_{2^+_{\mathrm{s}}}}$")

    for Nf in 1, 2:
        for beta, colour, marker in beta_colour_marker[Nf]:
            data_to_plot = hatted_data[
                (hatted_data.beta == beta)
                & ~(hatted_data.label.str.endswith("*"))
                & (hatted_data.Nf == Nf)
            ]
            if data_to_plot.value_ratio.isnull().all():
                continue

            ax.errorbar(
                data_to_plot.value_mpcac_mass_hat,
                data_to_plot.value_ratio,
                xerr=data_to_plot.uncertainty_mpcac_mass_hat,
                yerr=data_to_plot.uncertainty_ratio,
                color=colour,
                marker=marker,
                label=f"$\\beta={beta}$",
                ls="none",
            )

        ax.legend(
            ncol=3,
            columnspacing=1,
            handletextpad=0,
            borderpad=0,
            loc="lower center",
            frameon=False,
        )

        ax.set_xlim((0, None))
        ax.set_ylim((0, None))

        fig.tight_layout(pad=0.28)
        fig.savefig(filename.format(Nf=Nf))
        plt.close(fig)
