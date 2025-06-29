import matplotlib.pyplot as plt

from .common import format_ensembles_list, preliminary, ONE_COLUMN
from ..do_analysis import get_subdirectory_name
from ..plots import set_plot_defaults
from ..polyakov import fit_and_plot_polyakov_loops


ENSEMBLES = {
    1: ("DB4M13", "DB5M8", "DB6M9", "DB7M9"),
    2: ("Nf2DB0M6", "Nf2DB1M13", "Nf2DB2M7*"),
}
OUTPUT_DIR = "assets/plots"
FILENAME_BASE = "polyakov_Nf{Nf}"
CAPTION = r"Polyakov loop histograms, for the ensembles {ensembles}."


def do_plot(ensembles, ensemble_names_to_plot, filename_base):
    set_plot_defaults(linewidth=0.5, preliminary=preliminary)
    fig, axes = plt.subplots(
        len(ensemble_names_to_plot),
        sharex=True,
        figsize=(ONE_COLUMN, 8),
        squeeze=False,
    )

    for ensemble_name, ax_row in zip(ensemble_names_to_plot, axes):
        ax = ax_row[0]
        directory = get_subdirectory_name(ensembles[ensemble_name])
        ax.set_title(ensemble_name)
        try:
            fit_and_plot_polyakov_loops(
                "raw_data/" + directory + "/out_pl",
                num_bins=50,
                ax=ax,
                do_fit=False,
                label_axes=False,
            )
        except FileNotFoundError:
            print(
                f"Skipping plotting Polyakov loop for {ensemble_name} as file is missing."
            )
        ax.autoscale(axis="x")

    ax.set_xlabel(r"$\langle P_\mu\rangle$")
    axes[0][0].legend(loc="best", frameon=False, title=r"$\mu$")

    fig.tight_layout(pad=0.08)
    fig.savefig(OUTPUT_DIR + "/" + filename_base + ".pdf")
    plt.close(fig)


def do_caption(filename_base, ensembles, caption, figlabel):
    observables = {}

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
    for Nf in 1, 2:
        do_plot(ensembles, ENSEMBLES[Nf], FILENAME_BASE.format(Nf=Nf))
        do_caption(
            FILENAME_BASE.format(Nf=Nf), ENSEMBLES[Nf], CAPTION, f"polyakov-Nf{Nf}"
        )
