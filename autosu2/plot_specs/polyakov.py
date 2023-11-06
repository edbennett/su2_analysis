import numpy as np
import matplotlib.pyplot as plt

from .common import preliminary
from ..db import get_measurement_as_ufloat
from ..data import get_flows_from_raw
from ..do_analysis import get_subdirectory_name
from ..plots import set_plot_defaults
from ..polyakov import fit_and_plot_polyakov_loops


ENSEMBLES = 'DB1M9', 'DB2M7', 'DB3M8', 'DB4M11'
OUTPUT_DIR = 'final_plots'
FILENAME_BASE = 'polyakov'
CAPTION = r'Polyakov loop histograms, for the ensembles {ensembles}.'


def do_plot(ensembles, ensemble_names, filename_base):
    set_plot_defaults(linewidth=0.5, preliminary=preliminary)
    fig, axes = plt.subplots(
        len(ensemble_names),
        sharex=True,
        figsize=(3.5, 8),
        squeeze=False
    )

    for ensemble, ax_row in zip(ensemble_names, axes):
        ax = ax_row[0]
        directory = get_subdirectory_name(ensembles[ensemble])
        ax.set_title(ensemble)
        fit_and_plot_polyakov_loops(
            'raw_data/' + directory + '/out_pl',
            num_bins=50,
            ax=ax,
            do_fit=False,
            label_axes=False
        )
        ax.autoscale(axis='x')

    ax.set_xlabel(r'$\langle P_\mu\rangle$')
    axes[0][0].legend(loc='best', frameon=False, title=r'$\mu$')

    fig.tight_layout(pad=0.08)
    fig.savefig(OUTPUT_DIR + '/' + filename_base + '.pdf')
    plt.close(fig)


def do_caption(filename_base, ensembles, caption, figlabel):
    observables = {}

    if len(ensembles) > 2:
        observables['ensembles'] = '{}, and {}'.format(ensembles[:-1],
                                                       ensembles[-1])
    elif len(ensembles) == 2:
        observables['ensembles'] = '{} and {}'.format(*ensembles)
    elif len(ensembles) == 1:
        observables['ensembles'] = '{}'.format(*ensembles)

    caption = caption.format(**observables)

    with open(OUTPUT_DIR + '/' + filename_base + '.tex', 'w') as f:
        print(r'\begin{figure}', file=f)
        print(r'  \center', file=f)
        print(r'  \includegraphics{' + filename_base + r'}', file=f)
        print(r'  \caption{{{caption}}}'.format(caption=caption), file=f)
        print(r'  \label{{fig:{figlabel}}}'.format(figlabel=figlabel), file=f)
        print(r'\end{figure}', file=f)


def generate(data, ensembles):
    do_plot(ensembles, ENSEMBLES, FILENAME_BASE)
    do_caption(FILENAME_BASE, ENSEMBLES, CAPTION, 'polyakov')
