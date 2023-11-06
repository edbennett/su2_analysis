#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import plasma, ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .common import preliminary
from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, format_value_and_error
from ..do_analysis import get_subdirectory_name

def do_plot(data, filename=None, omega_min=None, omega_max=None, ensemble=None,
            fit_result=None, ax=None):

    if ensemble:
        omega_min = ensemble['measure_modenumber'].get(
            'plot_omega_min', omega_min
        )
        omega_max = ensemble['measure_modenumber'].get(
            'plot_omega_max', omega_max
        )

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
    l_min = round(data.window_length.min(), 2)
    l_max = round(data.window_length.max(), 2)

    # capsize=1 breaks multicolour plots so don't set this here
    if ax:
        ax_supplied = True
    else:
        ax_supplied = False
        set_plot_defaults(linewidth=0.5, capsize=0, preliminary=preliminary)
        fig, ax = plt.subplots(figsize=(3.5, 2))

    colour_norm = LogNorm(vmin=l_min, vmax=l_max)

    colours = plasma(colour_norm(data.window_length.values))
    colours[:, 3] -= data.badness.values.clip(max=100) / 100

    cbax = inset_axes(
        ax, width='70%', height='10%', loc='lower right', borderpad=1
    )
    cb = plt.colorbar(
        ScalarMappable(norm=colour_norm, cmap=plasma),
        cax=cbax,
        orientation='horizontal'
    )
    cbax.text(0.5, 1.75, r'$\Delta\Omega$', ha='center',
              transform=cbax.transAxes)
    cb.set_ticks((l_min, l_max))
    cb.minorticks_off()
    cb.set_ticklabels((f'{l_min}', f'{l_max}'))
    cbax.xaxis.set_ticks_position('top')
    cbax.xaxis.set_label_position('top')
    cbax.set_in_layout(False)

    if not ax_supplied:
        ax.set_xlabel(r'$\Omega_{\mathrm{LE}}$')
    elif ensemble:
        ax.text(
            0.125,
            0.1,
            ensemble['label'],
            horizontalalignment='center',
            verticalalignment='bottom',
            transform=ax.transAxes
        )
    ax.set_ylabel(r'$\gamma_*$')

    ax.scatter(data.omega_lower_bound.values, data.gamma_star.values,
               color=colours, linewidths=0)
    ax.errorbar(
        data.omega_lower_bound.values,
        data.gamma_star.values,
        yerr=data.gamma_star_error.values,
        linestyle='none',
        marker='None',
        ecolor=colours,
    )

    if fit_result and ensemble:
        fit_value, fit_error = fit_result
        ax.fill_between(
            (ensemble['measure_modenumber']['fit_omega_min'],
             ensemble['measure_modenumber']['fit_omega_max']),
            (fit_value - fit_error, fit_value - fit_error),
            (fit_value + fit_error, fit_value + fit_error),
            color='black',
            alpha=0.2,
            linestyle='None',
            linewidth=0
        )

    ax.set_ylim((0, 1.09))

    if not ax_supplied:
        fig.tight_layout(pad=0.08)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


def badness_to_weight(badness):
    return np.sinh(1) / np.sinh(1 - badness / 100)


def fit_modenumber_plateau(data, ensemble):
    '''
    Do a weighted fit of the mode number anomalous dimension plateau,
    accounting both for the spread of the points (gamma_star_error), and
    the level of trust in the points due to the number of successfull fits
    (badness).
    '''
    fit_limits = ensemble['measure_modenumber']
    data_to_fit = data[
        (data.omega_lower_bound > fit_limits['fit_omega_min'])
        & (data.omega_lower_bound < fit_limits['fit_omega_max'])
        & (data.window_length > fit_limits['fit_window_length_min'])
        & (data.window_length < fit_limits['fit_window_length_max'])
        & (data.badness < 100)
    ]

    fit_result, fit_variance = curve_fit(
        lambda x, gamma: gamma,
        data_to_fit.omega_lower_bound.values,
        data_to_fit.gamma_star.values,
        p0=(data_to_fit.gamma_star.values.mean(),),
        sigma=(
            data_to_fit.gamma_star_error.values
            * badness_to_weight(data_to_fit.badness)
        ),
        absolute_sigma=True
    )
    return fit_result[0], fit_variance[0, 0] ** 0.5


def tabulate(fit_results, ensembles):
    filename = 'modenumber_gamma.tex'
    columns = (
        'Ensemble',
        None,
        r'$\Omega_{\mathrm{LE}}^{\mathrm{min}}$',
        r'$\Omega_{\mathrm{LE}}^{\mathrm{max}}$',
        r'$\Delta \Omega_{\mathrm{min}}$',
        r'$\Delta \Omega_{\mathrm{max}}$',
        None,
        '$\gamma_*$'
    )
    table_content = []
    table_line = (
        '    {ensemble_name} & {omega_min} & {omega_max} & {len_min} '
        '& {len_max} & {gamma_star}'
    )

    for ensemble_name, gamma_star in fit_results.items():
        ensemble_parameters = ensembles[ensemble_name]['measure_modenumber']
        table_content.append(table_line.format(
            ensemble_name=ensemble_name,
            omega_min=ensemble_parameters['fit_omega_min'],
            omega_max=ensemble_parameters['fit_omega_max'],
            len_min=ensemble_parameters['fit_window_length_min'],
            len_max=ensemble_parameters['fit_window_length_max'],
            gamma_star = format_value_and_error(*gamma_star)
        ))

    generate_table_from_content(filename, table_content, columns)


def generate(data, ensembles):
    plot_filename = 'final_plots/modenumber.pdf'
    ensembles_to_plot = 'DB1M8', 'DB1M9', 'DB2M7', 'DB3M8', 'DB4M11'
    fit_results = {}

    # capsize=1 breaks multicolour plots so don't set this here
    set_plot_defaults(linewidth=0.5, capsize=0, preliminary=preliminary)
    fig, axes = plt.subplots(nrows=5, figsize=(3.5, 8))

    for ensemble_name, ax in zip(ensembles_to_plot, axes):
        modenumber_data = pd.read_csv(
            f'processed_data/{get_subdirectory_name(ensembles[ensemble_name])}'
            '/modenumber_fit.csv'
        ).dropna()
        modenumber_data['window_length'] = (
            modenumber_data.omega_upper_bound
            - modenumber_data.omega_lower_bound
        )
        fit_results[ensemble_name] = fit_modenumber_plateau(
            modenumber_data, ensembles[ensemble_name]
        )
        do_plot(
            modenumber_data,
            f'final_plots/modenumber_{ensemble_name}.pdf',
            ensemble={'label': ensemble_name, **ensembles[ensemble_name]},
            fit_result=fit_results[ensemble_name],
            ax=ax
        )
    axes[-1].set_xlabel(r'$\Omega_{\mathrm{LE}}$')
    fig.tight_layout(pad=0.08, h_pad=1)
    fig.savefig(plot_filename)
    plt.close(fig)

    tabulate(fit_results, ensembles)
