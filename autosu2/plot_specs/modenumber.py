#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import plasma, ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pandas as pd

from ..plots import set_plot_defaults
from ..do_analysis import get_subdirectory_name

def do_plot(data, filename, omega_min=None, omega_max=None):
    if omega_min is not None:
        data = data[data.omega_lower_bound >= omega_min]
    else:
        omega_min = data.omega_lower_bound.min()
    if omega_max is not None:
        data = data[data.omega_upper_bound <= omega_max]
    else:
        omega_max = data.omega_lower_bound.max()

    data = data.dropna(axis='index',
                       subset=('gamma_star', 'gamma_star_error'))

    # capsize=1 breaks multicolour plots so don't set this here
    set_plot_defaults(linewidth=0.5, capsize=0)
    fig, ax = plt.subplots(figsize=(3.5, 2))

    colour_norm = LogNorm(vmin=omega_min, vmax=omega_max)

    colours = plasma(colour_norm(data.omega_lower_bound.values))
    colours[:, 3] -= data.badness.values.clip(max=100) / 100

    cbax = inset_axes(ax, width='80%', height='10%', loc='lower center')
    cb = fig.colorbar(
        ScalarMappable(norm=colour_norm, cmap=plasma),
        cax=cbax,
        orientation='horizontal'
    )
    cbax.text(0.5, 1.75, 'Lower bound of window', ha='center',
              transform=cbax.transAxes)
    cb.set_ticks((omega_min, omega_max))
    cb.minorticks_off()
    cb.set_ticklabels((f'{omega_min}', f'{omega_max}'))
    cbax.xaxis.set_ticks_position('top')
    cbax.xaxis.set_label_position('top')
    cbax.set_in_layout(False)

    ax.set_xlabel('Window length')
    ax.set_ylabel(r'$\gamma_*$')

    data['window_length'] = data.omega_upper_bound - data.omega_lower_bound

    ax.scatter(data.window_length, data.gamma_star.values, color=colours)
    ax.errorbar(
        data.window_length.values,
        data.gamma_star.values,
        yerr=data.gamma_star_error.values,
        linestyle='none',
        marker='None',
        ecolor=colours,
    )
    ax.set_ylim((0, 1.09))

    fig.tight_layout(pad=0.08)
    fig.savefig(filename)
    plt.close(fig)


def generate(data, ensembles):
    ensembles_to_plot = (
        ('DB1M8', 0.03, 0.12),
        ('DB1M10', 0.03, 0.12),
        ('DB2M7', 0.03, 0.12),
        ('DB3M8', 0.04, 0.12),
        ('DB4M11', 0.04, 0.12)
    )

    for ensemble, omega_min, omega_max in ensembles_to_plot:
        data = pd.read_csv(
            f'processed_data/{get_subdirectory_name(ensembles[ensemble])}'
            '/modenumber_fit.csv'
        )
        do_plot(data, f'auxiliary_plots/modenumber_{ensemble}_invert.pdf',
                omega_min=omega_min, omega_max=omega_max)
