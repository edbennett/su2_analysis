import matplotlib.pyplot as plt

from ..db import get_dataframe
from ..plots import set_plot_defaults

from .fig10_11 import EFT_ENSEMBLES

FILENAME = 'final_plots/fig14.pdf'


def fig14(data, ensemble_names):
    set_plot_defaults(linewidth=0.5, markersize=1)

    subset = data[
        data.label.isin(EFT_ENSEMBLES)
        & data.observable.isin(
            ('g5_mass_hat_squared',
             'g5_renormalised_decay_const_hat_squared_continuum_corrected')
        )
    ]
    data_to_plot = subset.pivot(
        index='label', columns='observable', values=['value', 'uncertainty']
    )

    m2_value = data_to_plot.value.g5_mass_hat_squared
    m2_error = data_to_plot.uncertainty.g5_mass_hat_squared
    f2_value = (data_to_plot
                .value
                .g5_renormalised_decay_const_hat_squared_continuum_corrected)
    f2_error = (data_to_plot
                .uncertainty
                .g5_renormalised_decay_const_hat_squared_continuum_corrected)
    mf2_value = m2_value * f2_value
    mf2_error = (f2_value ** 2 * m2_error ** 2
                 + m2_value ** 2 * f2_error ** 2) ** 0.5

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.errorbar(m2_value, mf2_value, xerr=m2_error, yerr=mf2_error, fmt='b+')

    ax.set_xlim((0, 0.42))
    ax.set_ylim((0, 0.0054))
    ax.set_xlabel(r'$\hat{m}_{\mathrm{PS}}^2$')
    ax.set_ylabel(r'$\hat{m}_{\mathrm{PS}}^2 \hat{f}_{\mathrm{PS}}^2$')

    fig.tight_layout()
    fig.savefig(FILENAME)
    plt.close(fig)


def generate(data, ensembles):
    data = get_dataframe()
    fig14(data, EFT_ENSEMBLES)
