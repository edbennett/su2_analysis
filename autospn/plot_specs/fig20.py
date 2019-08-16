from os.path import join

import matplotlib.pyplot as plt
from matplotlib import ticker

from ..bootstrap import bootstrap_correlators, bootstrap_eff_masses
from ..fit_correlation_function import (
    get_target_correlator, channel_set_options, symmetries_options
)
from ..plots import (
    set_plot_defaults, COLOR_LIST, SYMBOL_LIST, do_eff_mass_plot
)
from .fig7_8_9 import CHANNEL_LABELS

FILENAME = 'final_plots/fig20.pdf'
CHANNELS = ('AV', 'S', 'V', 'PS')
ENSEMBLE = 'DB3M7'
ENSEMBLE_DIRECTORY = 'raw_data/nf2_FUN/36x28x28x28b7.2m-0.794'


def get_bootstrapped_eff_mass(correlator_filename, ensemble, channel):
    meson_parameters = ensemble['measure_mesons'][channel]
    target_correlator_sets, valence_masses = get_target_correlator(
        correlator_filename,
        channel_set_options[channel],
        ensemble['T'],
        ensemble['L'],
        symmetries_options[channel],
        meson_parameters['ensemble_selection'],
        ensemble['initial_configuration'],
        meson_parameters['configuration_separation'],
        from_raw=True
    )
    (bootstrap_mean_correlators, bootstrap_error_correlators,
     bootstrap_correlator_samples_set) = bootstrap_correlators(
         next(target_correlator_sets)
     )

    return bootstrap_eff_masses(bootstrap_correlator_samples_set)


def generate(data, ensembles):
    set_plot_defaults()
    fig = plt.figure(figsize=(4.5, 3))
    ax = fig.subplots()

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$a\,m_{\mathrm{eff}}$')

    ax.set_xlim((0, 18.5))
    ax.set_ylim((0.2, 1.3))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))

    for channel, colour, marker in zip(CHANNELS,
                                       COLOR_LIST[3::-1],
                                       SYMBOL_LIST[3::-1]):
        channel_label = CHANNEL_LABELS[channel]
        ensemble = ensembles[ENSEMBLE]
        meson_parameters = ensemble['measure_mesons'][channel_label]
        eff_mass, eff_mass_error = get_bootstrapped_eff_mass(
            join(ENSEMBLE_DIRECTORY, 'out_corr'),
            ensemble,
            channel_label
        )

        reading = data[data.label.eq(ENSEMBLE)
                       & data.observable.eq(f'{channel_label}_mass')]
        if len(reading) > 1:
            raise ValueError('Non-unique ensemble-mass combination')
        if len(reading) == 0:
            print('No data for this ensemble-mass combination')
            fit_mass = None
            fit_error = None
        else:
            fit_mass = float(reading.value)
            fit_error = float(reading.uncertainty)

        do_eff_mass_plot(
            eff_mass[0], eff_mass_error[0],
            tmin=meson_parameters['plateau_start'] - 0.5,
            tmax=meson_parameters['plateau_end'] - 0.5,
            m=fit_mass,
            m_error=fit_error,
            label=channel,
            colour=colour,
            marker=marker,
            ax=ax
        )

    ax.legend(frameon=False, loc=0)
    fig.tight_layout()
    fig.savefig(FILENAME)
    plt.close(fig)
