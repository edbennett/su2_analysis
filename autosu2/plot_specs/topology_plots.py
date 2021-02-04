import numpy as np
import matplotlib.pyplot as plt

from ..db import get_measurement_as_ufloat
from ..data import get_flows_from_raw
from ..do_analysis import get_subdirectory_name
from ..plots import set_plot_defaults
from ..Q import plot_history_and_histogram


ENSEMBLES = 'DB1M10', 'DB2M7', 'DB3M8', 'DB4M11'
OUTPUT_DIR = 'final_plots'
FILENAME_BASE = 'q_topology'
CAPTION = r'''
Topological charge histories (left), and histograms (right), for the ensembles
{ensembles}.'''


def do_plot(ensembles, ensemble_names, filename_base):
    set_plot_defaults(linewidth=0.5)
    fig, ax = plt.subplots(
        len(ensemble_names), 2,
        sharey='row',
        sharex='col',
        gridspec_kw={'width_ratios': [3, 1]},
        figsize=(3.5, 0.5 + 1.5 * len(ensemble_names)),
        squeeze=False
    )

    for ensemble, ax_row in zip(ensemble_names, ax):
        directory = get_subdirectory_name(ensembles[ensemble])
        ax_row[0].set_title(ensemble)
        trajectories, *_, Qs = get_flows_from_raw(
            'raw_data/' + directory + '/out_wflow'
        )
        plot_history_and_histogram(
            np.arange(len(Qs)), Qs,
            history_ax=ax_row[0], histogram_ax=ax_row[1], label_axes=False
        )

    ax[-1][0].set_xlabel('Trajectory')
    ax[-1][1].set_xlabel('Count')

    fig.tight_layout(pad=0.08)
    fig.savefig(OUTPUT_DIR + '/' + filename_base + '.pdf')
    plt.close(fig)


def do_caption(filename_base, ensembles, caption, figlabel):
    observables = {}
    for ensemble in ensembles:
        observables[f'{ensemble}_Q0'] = get_measurement_as_ufloat(
            {'label': ensemble}, 'fitted_Q0'
        )
        observables[f'{ensemble}_width'] = get_measurement_as_ufloat(
            {'label': ensemble}, 'Q_width'
        )
        tau_exp = get_measurement_as_ufloat(
            {'label': ensemble}, 'Q_tau_exp'
        )

        if tau_exp.n == 0:
            observables[f'{ensemble}_tauexp'] = r'\ll 1'
        else:
            observables[f'{ensemble}_tauexp'] = f'={tau_exp:.1uSL}'

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
    do_caption(FILENAME_BASE, ENSEMBLES, CAPTION, 'topcharge')
