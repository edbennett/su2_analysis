from math import ceil

import matplotlib.figure
import matplotlib.pyplot

from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, rc, close
from matplotlib.colors import XKCD_COLORS
from numpy import linspace, arange

from warnings import filterwarnings


REVTEX_FONT_SIZE = 10
COLOR_LIST = [
    XKCD_COLORS[f"xkcd:{colour}"]
    for colour in [
        "tomato red",
        "leafy green",
        "cerulean blue",
        "golden brown",
        "faded purple",
        "shocking pink",
        "pumpkin orange",
        "dusty teal",
        "red wine",
        "navy blue",
        "salmon",
    ]
]

SYMBOL_LIST = "+.*o^x1v2"

filterwarnings("ignore", category=UserWarning, module="matplotlib")


fig_original_init = Figure.__init__


def set_plot_defaults(
    fontsize=None, markersize=4, capsize=2, linewidth=1, preliminary=False
):
    if not fontsize:
        fontsize = REVTEX_FONT_SIZE

    font = {"size": fontsize}

    rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], **font})
    rc("text", usetex=True)
    rc("lines", linewidth=linewidth, markersize=markersize, markeredgewidth=linewidth)
    rc("errorbar", capsize=capsize)
    rc("axes", facecolor=(1, 1, 1, 0))

    if preliminary:

        def fig_patched_init(self, *args, **kwargs):
            fig_original_init(self, *args, **kwargs)
            prelim_fontsize = min(
                fontsize * 5, 50 / 4 * (self.get_size_inches()[0] * 0.9)
            )
            self.text(
                0.5,
                0.5,
                "PRELIMINARY",
                alpha=0.1,
                fontsize=prelim_fontsize,
                rotation=30,
                ha="center",
                va="center",
                zorder=-1,
            )

        matplotlib.figure.Figure.__init__ = fig_patched_init
    else:
        matplotlib.figure.Figure.__init__ = fig_original_init


def do_eff_mass_plot(
    masses,
    filename=None,
    ymin=None,
    ymax=None,
    tmin=None,
    tmax=None,
    m=None,
    ax=None,
    colour="red",
    marker="s",
    label=None,
):
    assert (filename is not None) or (ax is not None)

    if not ax:
        fig, ax = subplots()
        local_ax = True
    else:
        local_ax = False

    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = masses.T
    t_range_start = ceil(tmin)
    t_range_end = min(masses.T, ceil(tmax))

    timeslice, mass, mass_error = masses.plottable()

    ax.errorbar(
        timeslice[t_range_start:t_range_end],
        mass[t_range_start:t_range_end],
        yerr=mass_error[t_range_start:t_range_end],
        fmt=marker,
        color=colour,
        label=label,
    )

    if local_ax:
        ax.set_xlim((tmin, tmax + 1))
        if ymin is None and ymax is None:
            ax.autoscale(axis="y", tight=True)
        else:
            ax.set_ylim((ymin, ymax))

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$m_{\mathrm{eff}}$")

    if m is not None:
        if not tmin:
            tmin = 0
        if not tmax:
            tmax = masses.T
        # ax.plot((tmin, tmax), (m, m), color=colour)
        ax.fill_between(
            # tmin + 1 due to mixing of adjacent points when calculating
            # effective mass
            (tmin + 1, tmax),
            (m.value + m.dvalue, m.value + m.dvalue),
            (m.value - m.dvalue, m.value - m.dvalue),
            facecolor=colour,
            alpha=0.4,
        )

    if local_ax:
        fig.tight_layout()
        fig.savefig(filename)
        close(fig)


def do_correlator_plot(
    correlator,
    filename,
    channel_latex,
    fit_function=None,
    fit_params=None,
    fit_legend="",
    t_lowerbound=None,
    t_upperbound=None,
    corr_lowerbound=None,
    corr_upperbound=None,
):
    if not t_lowerbound:
        t_lowerbound = 0
    if not t_upperbound:
        t_upperbound = correlator.T - 1

    fig, ax = subplots()

    timeslice, correlator_value, correlator_error = correlator.plottable()
    ax.errorbar(
        timeslice, correlator_value, yerr=correlator_error, fmt="o", label="Data"
    )
    ax.set_xlim((t_lowerbound, t_upperbound))
    ax.set_ylim((corr_lowerbound, corr_upperbound))
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$C_{" f"{channel_latex}" r"}(t)$")

    if fit_function:
        if not fit_params:
            fit_params = {}
        t_range = linspace(t_lowerbound, t_upperbound, 1000)
        ax.plot(t_range, fit_function(t=t_range, **fit_params), label=fit_legend)
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
    close(fig)
