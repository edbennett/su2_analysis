from itertools import product

import matplotlib.pyplot as plt

from .common import beta_colour_marker, preliminary

from ..plots import set_plot_defaults, SYMBOL_LIST
from ..derived_observables import merge_no_w0

OBSERVABLES = "mass", "decay_const"
CHANNELS = "g5", "gk", "g5gk", "id"
OBSERVABLE_LABELS = "m", "f"
CHANNEL_LABELS = r"\gamma_5", r"\gamma_k", r"\gamma_5\gamma_k", r"1"


def do_plot(data, Nf=1):
    set_plot_defaults(preliminary=preliminary)
    filename = f"auxiliary_plots/spectra_Nf{Nf}.pdf"
    fig, axes = plt.subplots(
        ncols=2,
        nrows=len(beta_colour_marker[Nf]),
        figsize=(10, 2 + 4 * len(beta_colour_marker[Nf])),
        squeeze=False,
    )

    merged_data = merge_no_w0(
        data[data.Nf == Nf],
        [
            f"{channel}_{observable}"
            for channel in CHANNELS
            for observable in OBSERVABLES
        ]
        + ["mpcac_mass"],
    )

    for column, (observable, observable_label) in enumerate(
        zip(OBSERVABLES, OBSERVABLE_LABELS)
    ):
        for row, (beta, _, _) in enumerate(beta_colour_marker[Nf]):
            current_data = merged_data[merged_data.beta == beta]
            axes[row][column].set_title(f"$\\beta = {beta}$")
            axes[row][column].set_xlabel(r"$m_{\mathrm{PCAC}}$")
            axes[row][column].set_ylabel(f"$a{observable_label}_X$")

            for key_index, (channel, channel_label) in enumerate(
                zip(CHANNELS, CHANNEL_LABELS)
            ):
                axes[row][column].errorbar(
                    current_data["value_mpcac_mass"],
                    current_data[f"value_{channel}_{observable}"],
                    xerr=current_data["uncertainty_mpcac_mass"],
                    yerr=current_data[f"uncertainty_{channel}_{observable}"],
                    color=f"C{key_index}",
                    fmt=SYMBOL_LIST[key_index],
                    label=f"${channel_label}$",
                )

            axes[row][column].set_xlim((0, None))
            axes[row][column].set_ylim((0, None))

    axes[-1][1].legend(loc="lower right", frameon=False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def generate(data, ensembles):
    do_plot(data, Nf=1)
    do_plot(data, Nf=2)
