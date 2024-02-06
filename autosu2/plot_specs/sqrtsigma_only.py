#!/usr/bin/env python3

import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import (
    beta_colour_marker,
    critical_ms,
    preliminary,
)


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.5, preliminary=preliminary)
    Nf = 1

    columns_to_hat = ["mpcac_mass", "sqrtsigma"]

    hatted_data = merge_and_hat_quantities(data, columns_to_hat)
    hatted_data["glue_code"] = hatted_data.apply(
        lambda row: (
            "New"
            if "torelon" in ensembles.get(row.label, {}).get("measure_glueballs", {})
            else "Old"
        ),
        axis="columns",
    )

    fig, ax = plt.subplots()

    for (beta, colour, _), m_c in zip(beta_colour_marker[Nf], critical_ms[Nf]):
        for glue_code, marker in ("Old", "o"), ("New", "^"):
            data_to_plot = hatted_data[
                (hatted_data.beta == beta)
                & ~(hatted_data.label.str.endswith("*"))
                & (hatted_data.Nf == 1)
                & (hatted_data.glue_code == glue_code)
            ]
            ax.errorbar(
                data_to_plot.value_mpcac_mass_hat,
                data_to_plot.value_sqrtsigma_hat,
                xerr=data_to_plot.uncertainty_mpcac_mass_hat,
                yerr=data_to_plot.uncertainty_sqrtsigma_hat,
                color=colour,
                marker=marker,
                ls="none",
            )

    for glue_code, marker in ("Old", "o"), ("New", "^"):
        ax.scatter([-1], [-1], marker=marker, color="black", label=glue_code)

    ax.legend(
        loc="upper left",
        frameon=False,
        handletextpad=0,
        ncol=2,
        columnspacing=0.3,
        borderaxespad=0.2,
        fontsize="small",
    )
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    fig.savefig("auxiliary_plots/sqrtsigma_Nf1.pdf")
    plt.close(fig)
