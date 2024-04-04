from collections import namedtuple

from numpy import nan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import beta_colour_marker, preliminary

ALPHA = 0.3
R_value = namedtuple("R_value", ["centre", "lower", "upper"])


def plot_points(data, beta, colour, marker, ax):
    ax.set_ylabel(r"$\frac{M_{2^{++}}}{M_{0^{++}}}$")
    ax.errorbar(
        data["value_A1++_mass"] * data.L,
        data.value_R,
        xerr=data["uncertainty_A1++_mass"] * data.L,
        yerr=data.uncertainty_R,
        color=colour,
        marker=marker,
        label=f"$\\beta={beta}$",
        ls="none",
    )
    ax.text(0.03, 0.05, f"$\\beta={beta}$", transform=ax.transAxes)


def shade_prediction(colour, R, ax):
    # ax.set_xlim((0, None))
    xlim = ax.get_xlim()
    _, ymax = ax.get_ylim()
    ymin = 0

    # SU(2) pure gauge
    ax.axhline(1.44, dashes=(3, 3))

    ax.axhline(R.centre, color=colour)
    if R.upper:
        R_upper = R.upper
        ymax = None
    elif ymax < R.centre:
        R_upper = R.centre + 0.2 * (ymax - ymin)
        ymax = R_upper
    else:
        R_upper = ymax

    if R.lower:
        R_lower = R.lower
        ymin = 0
    elif ymin > R.centre:
        R_lower = R.centre - 0.2 * (ymax - ymin)
        ymin = R_lower
    else:
        R_lower = ymin

    ax.fill_between(
        xlim, (R_lower, R_lower), (R_upper, R_upper), color=colour, alpha=ALPHA
    )
    ax.set_xlim(xlim)
    ax.set_ylim((ymin, ymax))


def plot_single(data, Nf, predicted_Rs, filename):
    num_subplots = len(beta_colour_marker[Nf])
    fig, axes_2d = plt.subplots(nrows=num_subplots, figsize=(3.5, 2.5 + num_subplots * 1.5), sharex=True, squeeze=False)
    axes = axes_2d.ravel()
    hatted_data = merge_and_hat_quantities(
        data, ("A1++_mass", "E++_mass", "T2++_mass", "spin12_mass", "sqrtsigma")
    )
    hatted_data["value_R"] = (
        hatted_data["value_E++_mass"] / hatted_data["value_A1++_mass"]
    )
    hatted_data["uncertainty_R"] = (
        hatted_data["uncertainty_E++_mass"] ** 2 / hatted_data["value_A1++_mass"] ** 2
        + hatted_data["value_E++_mass"] ** 2
        * hatted_data["uncertainty_A1++_mass"] ** 2
        / hatted_data["value_A1++_mass"] ** 2
    ) ** 0.5
    axes[-1].set_xlabel(r"$L M_{0^{++}}$")

    for (beta, colour, marker), ax in zip(beta_colour_marker[Nf], axes):
        data_to_plot = hatted_data[
            (hatted_data.beta == beta) & (hatted_data.Nf == Nf)
            # R ratio is interesting at smaller volumes too
            # & ~(hatted_data.label.str.endswith('*'))
        ]
        plot_points(data_to_plot, beta, colour, marker, ax)

    for (_, colour, _), R, ax in zip(beta_colour_marker[1], predicted_Rs, axes):
        shade_prediction(colour, R, ax)

    legend_elements = [
        axes[-1].errorbar(
            [nan],
            [nan],
            yerr=[nan],
            xerr=[nan],
            ls="None",
            marker="x",
            color="black",
            label="Data",
        ),
        Line2D([0], [0], dashes=(3, 3), label="Pure SU(2)"),
        Patch(facecolor="black", alpha=ALPHA, label="Gauge-gravity model"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        frameon=False,
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.tight_layout(pad=0.28, rect=(0, 0.04, 1, 1))
    fig.savefig(filename, transparent=True)
    plt.close(fig)


def generate(data, ensembles):
    set_plot_defaults(markersize=4, capsize=1.0, linewidth=0.5, preliminary=preliminary)
    filename = "final_plots/Nf{Nf}_R_ratio.pdf"

    predicted_Rs_Nf1 = (
        R_value(7.1511, 4.7955, None),
        R_value(5.3377, 3.5782, None),
        R_value(3.6911, 3.3323, 4.1790),
        R_value(3.1640, 2.9379, 3.4791),
    )
    plot_single(data, 1, predicted_Rs_Nf1, filename.format(Nf=1))

    predicted_Rs_Nf2 = []
    plot_single(data, 2, predicted_Rs_Nf2, filename.format(Nf=2))
