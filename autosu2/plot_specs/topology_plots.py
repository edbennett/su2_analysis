from flow_analysis.readers import readers
import matplotlib.pyplot as plt

from .common import format_ensembles_list, preliminary
from ..db import get_measurement_as_ufloat
from ..do_analysis import get_subdirectory_name
from ..plots import set_plot_defaults
from ..Q import plot_history_and_histogram


ENSEMBLES = "DB4M13", "DB5M8", "DB6M9", "DB7M10"
OUTPUT_DIR = "final_plots"
FILENAME_BASE = "q_topology"
CAPTION = r"""
Topological charge histories (left), and histograms (right), for the ensembles
{ensembles}."""


def do_plot(ensembles, ensemble_names, filename_base):
    set_plot_defaults(linewidth=0.5, preliminary=preliminary)
    fig, ax = plt.subplots(
        len(ensemble_names),
        2,
        sharey="row",
        sharex="col",
        gridspec_kw={"width_ratios": [3, 1]},
        figsize=(3.5, 0.5 + 1.5 * len(ensemble_names)),
        squeeze=False,
        layout="constrained",
    )

    for ensemble, ax_row in zip(ensemble_names, ax):
        directory = get_subdirectory_name(ensembles[ensemble])
        ax_row[0].set_title(ensemble)
        reader_name = ensembles[ensemble].get("measure_gflow")
        if reader_name is True:
            reader_name = "hirep"
        if not reader_name:
            continue

        flows = readers[reader_name]("raw_data/" + directory + "/out_wflow")
        plot_history_and_histogram(
            flows, history_ax=ax_row[0], histogram_ax=ax_row[1], label_axes=False, count_axis="relative"
        )

    ax[-1][0].set_xlabel("Trajectory")
    ax[-1][1].set_xlabel("Proportion")

    fig.savefig(OUTPUT_DIR + "/" + filename_base + ".pdf")
    plt.close(fig)


def do_caption(filename_base, ensembles, caption, figlabel):
    observables = {}
    for ensemble in ensembles:
        observables[f"{ensemble}_Q0"] = get_measurement_as_ufloat(
            {"label": ensemble}, "fitted_Q0"
        )
        observables[f"{ensemble}_width"] = get_measurement_as_ufloat(
            {"label": ensemble}, "Q_width"
        )
        tau_exp = get_measurement_as_ufloat({"label": ensemble}, "Q_tau_exp")

        if tau_exp.n == 0:
            observables[f"{ensemble}_tauexp"] = r"\ll 1"
        else:
            observables[f"{ensemble}_tauexp"] = f"={tau_exp:.1uSL}"

    observables["ensembles"] = format_ensembles_list(ensembles)
    caption = caption.format(**observables)

    with open(OUTPUT_DIR + "/" + filename_base + ".tex", "w") as f:
        print(r"\begin{figure}", file=f)
        print(r"  \center", file=f)
        print(r"  \includegraphics{" + filename_base + r"}", file=f)
        print(r"  \caption{{{caption}}}".format(caption=caption), file=f)
        print(r"  \label{{fig:{figlabel}}}".format(figlabel=figlabel), file=f)
        print(r"\end{figure}", file=f)


def generate(data, ensembles):
    do_plot(ensembles, ENSEMBLES, FILENAME_BASE)
    do_caption(FILENAME_BASE, ENSEMBLES, CAPTION, "topcharge")
