from numpy import nan
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import (
    beta_colour_marker, critical_ms, channel_labels, add_figure_key
)

use_pcac = True

plots = [
    {
        'filename': 'final_plots/decayconst.pdf',
        'figsize': (3.5, 2.5),
        'subplots': [
            {
                'ylabel': r'$w_0 f$',
                'series': [
                    {
                        'channel': 'g5',
                        'quantity': 'decay_const'
                    }
                ]
            }
        ]
    },
    {
        'filename': 'final_plots/masses.pdf',
        'figsize': (3.5, 5),
        'subplots': [
            {
                'ylabel': r'$w_0 M$',
                'series': [
                    {
                        'channel': 'g5',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'g5gk',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'id',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'App',
                        'quantity': 'mass'
                    },
                ]
            },
            {
                'ylabel': r'$w_0 M$',
                'series': [
                    {
                        'channel': 'Epp',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'gk',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'spin12',
                        'quantity': 'mass'
                    },
                    {
                        'channel': 'sqrtsigma',
                        'quantity': ''
                    },
                ]
            }
        ]
    }
]


def do_plot(hatted_data, plot_spec):
    fig, axes = plt.subplots(
        nrows=len(plot_spec['subplots']),
        sharex=True,
        figsize=plot_spec['figsize']
    )
    if len(plot_spec['subplots']) == 1:
        axes = [axes]

    if use_pcac:
        axes[-1].set_xlabel(r'$w_0 m_{\mathrm{PCAC}}$')
    else:
        axes[-1].set_xlabel(r'$w_0 (m - m_c)$')

    markers = '.', 'x', '+', '^', 'v', '1', '2', '*'

    for subplot, ax in zip(plot_spec['subplots'], axes):
        ax.set_ylabel(subplot['ylabel'])

        for (beta, colour, _), m_c in zip(beta_colour_marker, critical_ms):
            data_to_plot = hatted_data[
                (hatted_data.beta == beta) &
                ~(hatted_data.label.str.endswith('*'))
            ]
            if use_pcac:
                mhat = data_to_plot.value_mpcac_mass_hat
                mhat_err = data_to_plot.uncertainty_mpcac_mass_hat
            else:
                mhat = (data_to_plot.m - m_c) * data_to_plot.value_w0
                mhat_err = (data_to_plot.m - m_c) * data_to_plot.uncertainty_w0

            for series, marker in zip(subplot['series'], markers):
                infix = '{channel}_{quantity}'.format(**series).strip('_')
                if f'value_{infix}_hat' in data_to_plot:
                    ax.errorbar(mhat,
                                data_to_plot[f'value_{infix}_hat'],
                                xerr=mhat_err,
                                yerr=data_to_plot[f'uncertainty_{infix}_hat'],
                                color=colour,
                                marker=marker,
                                ls='none')

        for series, marker in zip(subplot['series'], markers):
            ax.scatter([-1], [-1], marker=marker, color='black',
                       label=f'{channel_labels[series["channel"]]}')

        # Make room for legend
        ax.set_ylim((0, None))
        ylim = list(ax.get_ylim())
        ylim[1] *= 1.25
        ax.set_ylim(ylim)
        ax.legend(loc='upper left', frameon=False, handletextpad=0,
                  ncol=2, columnspacing=0.3, borderaxespad=0.2,
                  fontsize='small')

    ax.set_xlim((0, None))
    add_figure_key(fig, markers=False)

    fig.tight_layout(
        pad=0.12,
        h_pad=0.5,
        rect=(0.05, 0, 1, 1 - 0.3 / plot_spec['figsize'][1])
    )
    fig.savefig(plot_spec['filename'])
    plt.close(fig)


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.5)

    columns_to_hat = ['mpcac_mass'] + [
        '{channel}_{quantity}'.format(**series).strip('_')
        for plot_spec in plots
        for subplot in plot_spec.get('subplots', [])
        for series in subplot.get('series', [])
    ]

    hatted_data = merge_and_hat_quantities(data, columns_to_hat)

    for plot_spec in plots:
        do_plot(hatted_data, plot_spec)
