from itertools import product

import matplotlib.pyplot as plt

from ..plots import set_plot_defaults, SYMBOL_LIST
from ..derived_observables import merge_and_hat_quantities

OBSERVABLES = 'mass', 'decay_const'
CHANNELS = 'g5', 'g5gk', 'id'
BETAS = 2.05, 2.1, 2.15, 2.2
OBSERVABLE_LABELS = 'm', 'f'
CHANNEL_LABELS = r'\gamma_5', r'\gamma_5\gamma_k', r'1'


def generate(data, ensembles):
    set_plot_defaults()
    filename = f'auxiliary_plots/spectra.pdf'
    fig, axes = plt.subplots(ncols=2, nrows=4, figsize=(10, 18))

    merged_data = merge_and_hat_quantities(
        data,
        [f'{channel}_{observable}' 
         for channel in CHANNELS 
         for observable in OBSERVABLES]
        + ['mpcac_mass']
    )

    for column, (observable, observable_label) in enumerate(
            zip(OBSERVABLES, OBSERVABLE_LABELS)
    ):
        for row, beta in enumerate(BETAS):
            current_data = merged_data[merged_data.beta == beta]
            axes[row][column].set_title(f'$\\beta = {beta}$')
            axes[row][column].set_xlabel(r'$m_{\mathrm{PCAC}}$')
            axes[row][column].set_ylabel(f'$a{observable_label}_X$')

            for key_index, (channel, channel_label) in enumerate(
                    zip(CHANNELS, CHANNEL_LABELS)
            ):
                axes[row][column].errorbar(
                    current_data['value_mpcac_mass'],
                    current_data[f'value_{channel}_{observable}'],
                    xerr=current_data['uncertainty_mpcac_mass'],
                    yerr=current_data[f'uncertainty_{channel}_{observable}'],
                    color=f'C{key_index}',
                    fmt=SYMBOL_LIST[key_index],
                    label=f'${channel_label}$'
                )

            axes[row][column].set_xlim((0, None))
            axes[row][column].set_ylim((0, None))

    axes[3][1].legend(loc='lower right', frameon=False)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
