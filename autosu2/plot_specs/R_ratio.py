from collections import namedtuple
from functools import cache
import logging

from numpy import isnan, nan
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
from scipy.interpolate import interp1d

from ..plots import set_plot_defaults
from ..derived_observables import merge_no_w0

from .common import beta_colour_marker, preliminary
from .fshs import gammastar_fshs

ALPHA = 0.3
R_value = namedtuple("R_value", ["centre", "lower", "upper"])


@cache
def get_interpolator():
    data = pd.read_csv("external_data/sigmamodel.csv")
    raw_interpolator = interp1d(
        data.gammastar - 1, data.R_ratio, bounds_error=False, kind="cubic"
    )

    def interpolator(gammastar):
        if hasattr(gammastar, "__len__"):
            return [R if not isnan(R) else None for R in raw_interpolator(gammastar)]
        else:
            R = raw_interpolator(gammastar)
            return R if not isnan(R) else None

    return interpolator


def get_band(merged_data, Nf, beta):
    gammastar = gammastar_fshs(merged_data, Nf, beta)
    interpolator = get_interpolator()
    return R_value(
        *interpolator(
            [
                gammastar.nominal_value,
                gammastar.nominal_value - gammastar.std_dev,
                gammastar.nominal_value + gammastar.std_dev,
            ]
        )
    )


def plot_points(data, beta, marker, ax, mapper):
    ax.set_ylabel(r"$\frac{M_{2^{++}}}{M_{0^{++}}}$")
    for _, datum in data.iterrows():
        ax.errorbar(
            datum["value_A1++_mass"] * datum.L,
            datum.value_R,
            xerr=datum["uncertainty_A1++_mass"] * datum.L,
            yerr=datum.uncertainty_R,
            color=mapper.to_rgba(datum["value_A1++_mass"]),
            marker=marker,
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


def plot_single(data, Nf, filename):
    merged_data = merge_no_w0(
        data,
        (
            "mpcac_mass",
            "g5_mass",
            "E++_mass",
            "T2++_mass",
            "A1++_mass",
            "2++_mass",
            "spin12_mass",
            "sqrtsigma",
        ),
    )
    merged_data["value_R"] = (
        merged_data["value_2++_mass"] / merged_data["value_A1++_mass"]
    )
    merged_data["uncertainty_R"] = (
        merged_data["uncertainty_2++_mass"] ** 2 / merged_data["value_A1++_mass"] ** 2
        + merged_data["value_2++_mass"] ** 2
        * merged_data["uncertainty_A1++_mass"] ** 2
        / merged_data["value_A1++_mass"] ** 2
    ) ** 0.5

    betas = sorted(
        set(merged_data[merged_data.Nf == Nf].dropna(subset=["value_R"]).beta)
    )
    num_subplots = len(betas)

    if num_subplots == 0:
        logging.warning(f"No data to plot for R ratio for {Nf=}")
        return

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap="viridis")

    fig, axes_2d = plt.subplots(
        nrows=num_subplots,
        figsize=(3.5, 2.5 + num_subplots * 1.5),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    axes = axes_2d.ravel()

    axes[-1].set_xlabel(r"$L M_{0^{++}}$")
    axes[-1].set_xlim(2, 13)

    for target_beta, ax in zip(betas, axes):
        for beta, _, marker in beta_colour_marker[Nf]:
            if target_beta == beta:
                break
        else:
            raise ValueError(f"Don't have markers for {beta=}")

        data_to_plot = merged_data[
            (merged_data.beta == beta) & (merged_data.Nf == Nf)
            # R ratio is interesting at smaller volumes too
            # & ~(merged_data.label.str.endswith('*'))
        ]
        plot_points(data_to_plot, beta, marker, ax, mapper)

        predicted_R = get_band(merged_data, Nf, beta)
        shade_prediction("grey", predicted_R, ax)

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

    fig.colorbar(mapper, ax=axes[0], location="top", label="$aM_{0^{++}}$")

    fig.legend(
        handles=legend_elements,
        loc="outside lower center",
        frameon=False,
        ncol=3,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    fig.savefig(filename, transparent=True)
    plt.close(fig)


def generate(data, ensembles):
    set_plot_defaults(markersize=4, capsize=1.0, linewidth=0.5, preliminary=preliminary)

    for Nf in 1, 2:
        filename = f"assets/plots/R_ratio_Nf{Nf}.pdf"
        plot_single(data, Nf, filename)
