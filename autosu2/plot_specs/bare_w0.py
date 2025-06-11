import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..w0 import DEFAULT_W0

from .common import beta_colour_marker, TWO_COLUMN


def generate(data, ensembles):
    set_plot_defaults()

    filename = "auxiliary_plots/bare_w0.pdf"

    fig, axes = plt.subplots(
        ncols=2, layout="constrained", figsize=(TWO_COLUMN, 3.5), sharey=True
    )

    for ax, (Nf, betas) in zip(axes, beta_colour_marker.items()):
        for beta, colour, marker in betas:
            subset = data[
                (data.Nf == Nf)
                & (data.beta == beta)
                & (data.observable == "w0c")
                & (data.free_parameter == DEFAULT_W0)
                & ~(data.label.str.endswith("*"))
            ]
            ax.errorbar(
                subset.m,
                subset.value,
                yerr=subset.uncertainty,
                color=colour,
                marker=marker,
                ls="none",
                label=f"{beta}",
            )

        ax.set_title(f"$N_{{\\mathrm{{f}}}} = {Nf}$")
        ax.set_xlabel(r"$am_{{\mathrm{{f}}}}$")

    axes[0].set_ylabel(r"$w_0 / a$")
    fig.legend(loc="outside lower center", title=r"$\beta$", ncols=8)

    fig.savefig(filename)
