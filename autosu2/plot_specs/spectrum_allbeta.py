import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import (
    beta_colour_marker,
    critical_ms,
    channel_labels,
    add_figure_key,
    preliminary,
    ONE_COLUMN,
    TWO_COLUMN,
)

use_pcac = True

plots = [
    {
        "filename": "assets/plots/decayconst_Nf{Nf}.pdf",
        "figsize": (ONE_COLUMN, 3.0),
        "subplots": [
            {
                "ylabel": r"$w_0 f$",
                "series": [
                    {"channel": "g5", "quantity": "decay_const"},
                    {"channel": "g5gk", "quantity": "decay_const"},
                    {"channel": "gk", "quantity": "decay_const"},
                    {"channel": "id", "quantity": "decay_const"},
                ],
            }
        ],
        "legend_rows": 2,
        "ylim": (0.0, 2.0),
    },
    {
        "filename": "assets/plots/masses_Nf{Nf}.pdf",
        "figsize": (TWO_COLUMN, 3.5),
        "subplots": [
            {
                "ylabel": r"$w_0 M$",
                "series": [
                    {"channel": "g5", "quantity": "mass"},
                    {"channel": "g5gk", "quantity": "mass"},
                    {"channel": "id", "quantity": "mass"},
                    {"channel": "A1++", "quantity": "mass"},
                ],
            },
            {
                "series": [
                    {"channel": "2++", "quantity": "mass"},
                    {"channel": "gk", "quantity": "mass"},
                    {"channel": "spin12", "quantity": "mass"},
                    {"channel": "sqrtsigma", "quantity": ""},
                ]
            },
        ],
        "ylim": (0.0, 4.8),
    },
]


def do_plot(hatted_data, plot_spec, Nf=1):
    fig, axes = plt.subplots(
        ncols=len(plot_spec["subplots"]),
        sharey=True,
        figsize=plot_spec["figsize"],
        layout="constrained",
    )
    if len(plot_spec["subplots"]) == 1:
        axes = [axes]

    markers = ".", "x", "*", "^", "v", "1", "2", "+"

    for subplot, ax in zip(plot_spec["subplots"], axes):
        if use_pcac:
            ax.set_xlabel(r"$w_0 m_{\mathrm{PCAC}}$")
        else:
            ax.set_xlabel(r"$w_0 (m - m_c)$")
        if "ylabel" in subplot:
            ax.set_ylabel(subplot["ylabel"])

        for (beta, colour, _), m_c in zip(beta_colour_marker[Nf], critical_ms[Nf]):
            data_to_plot = hatted_data[
                (hatted_data.beta == beta)
                & ~(hatted_data.label.str.endswith("*"))
                & (hatted_data.Nf == Nf)
            ]
            if use_pcac:
                mhat = data_to_plot.value_mpcac_mass_hat
                mhat_err = data_to_plot.uncertainty_mpcac_mass_hat
            else:
                mhat = (data_to_plot.m - m_c) * data_to_plot.value_w0
                mhat_err = (data_to_plot.m - m_c) * data_to_plot.uncertainty_w0

            for series, marker in zip(subplot["series"], markers):
                infix = "{channel}_{quantity}".format(**series).strip("_")
                y_values = data_to_plot[f"value_{infix}_hat"]
                y_errors = data_to_plot[f"uncertainty_{infix}_hat"]
                mask = (y_errors / y_values) < 1.0
                if f"value_{infix}_hat" in data_to_plot:
                    ax.errorbar(
                        mhat[mask],
                        y_values[mask],
                        xerr=mhat_err[mask],
                        yerr=y_errors[mask],
                        color=colour,
                        marker=marker,
                        ls="none",
                    )

        for series, marker in zip(subplot["series"], markers):
            ax.scatter(
                [-1],
                [-1],
                marker=marker,
                color="black",
                label=f'{channel_labels[series["channel"]]}',
            )

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

    if (ylim := plot_spec.get("ylim")) is None:
        # Make room for legend
        axes[0].set_ylim((0, None))
        ylim = list(ax.get_ylim())
        ylim[1] *= 1.2

    axes[0].set_ylim(ylim)

    legend_rows = plot_spec.get("legend_rows", 1)
    add_figure_key(fig, markers=False, Nf=Nf, nrow=legend_rows)

    fig.savefig(plot_spec["filename"].format(Nf=Nf), transparent=True)
    plt.close(fig)


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.5, preliminary=preliminary)

    columns_to_hat = ["mpcac_mass"] + [
        "{channel}_{quantity}".format(**series).strip("_")
        for plot_spec in plots
        for subplot in plot_spec.get("subplots", [])
        for series in subplot.get("series", [])
    ]

    hatted_data = merge_and_hat_quantities(data, columns_to_hat)

    for plot_spec in plots:
        for Nf in 1, 2:
            do_plot(hatted_data, plot_spec, Nf=Nf)
