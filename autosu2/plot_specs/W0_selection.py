import matplotlib.pyplot as plt
from numpy import nan

from .common import preliminary
from ..plots import set_plot_defaults, COLOR_LIST

ENSEMBLES = ("DB2M1", "DB2M2", "DB2M3", "DB2M4", "DB2M5", "DB2M6", "DB2M7")
XS = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, ensembles):
    filename = "auxiliary_plots/W0_selection.pdf"

    set_plot_defaults(preliminary=preliminary)

    fig, (t_ax, w_ax) = plt.subplots(figsize=(3.5, 6), nrows=2, sharex=True)

    w_ax.set_xlabel(r"$am_0$")
    t_ax.set_ylabel(r"$\sqrt{8t_0}/a$")
    w_ax.set_ylabel(r"$w_0/a$")

    for label, symbol in (("Plaquette", "+"), ("Clover", ".")):
        t_ax.plot([nan], [nan], symbol, label=label, color="black")

    for X, colour in zip(XS, COLOR_LIST):
        for ax, observable in ((t_ax, "s8t0"), (w_ax, "w0")):
            for stencil, symbol in (("p", "+"), ("c", ".")):
                data_to_plot = data[
                    data.label.isin(ENSEMBLES)
                    & (data.observable == f"{observable}{stencil}")
                    & (data.free_parameter == X)
                ]
                ax.errorbar(
                    data_to_plot.m,
                    data_to_plot.value,
                    fmt=symbol,
                    yerr=data_to_plot.uncertainty,
                    color=colour,
                )
        t_ax.errorbar(
            [nan],
            [nan],
            yerr=nan,
            fmt=".",
            markersize=0,
            label=r"${:.02}$".format(X),
            color=colour,
            capsize=3,
        )

    m_min_t, m_max_t = t_ax.get_xlim()
    m_min_w, m_max_w = w_ax.get_xlim()
    combined_xlim = min(m_min_t, m_min_w), max(m_max_t, m_max_w)

    t_ax.set_xlim(combined_xlim)
    w_ax.set_xlim(combined_xlim)
    plt.setp(t_ax, xticklabels=[])

    fig.legend(
        loc="lower center", frameon=False, ncol=5, handletextpad=0, columnspacing=0.4
    )

    plt.tight_layout(pad=0.08, h_pad=0.5, rect=(0, 0.1, 1, 1))
    plt.savefig(filename)
    plt.close(fig)
