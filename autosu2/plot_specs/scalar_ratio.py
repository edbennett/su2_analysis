#!/usr/bin/env python

from numpy import nan
import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import beta_colour_marker


def generate(data, ensembles):
    set_plot_defaults(markersize=2, capsize=1, linewidth=0.5)

    filename = f'final_plots/scalar_ratio.pdf'
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    hatted_data = merge_and_hat_quantities(
        data,
        ('App_mass', 'g5_mass', 'mpcac_mass')
    )
    hatted_data['value_ratio'] = (
        hatted_data.value_App_mass / hatted_data.value_g5_mass
    )
    hatted_data['uncertainty_ratio'] = (
        hatted_data.uncertainty_App_mass ** 2 / hatted_data.value_g5_mass ** 2
        + hatted_data.value_App_mass ** 2
        * hatted_data.uncertainty_g5_mass ** 2
        / hatted_data.value_g5_mass ** 2
    ) ** 0.5
    ax.set_xlabel(r'$w_0 m_{\mathrm{PCAC}}$')
    ax.set_ylabel(r'$\frac{m_{0^{++}}}{m_{2^+_{\mathrm{s}}}}$')

    for beta, colour, marker in beta_colour_marker:
        data_to_plot = hatted_data[
            (hatted_data.beta == beta)
            & ~(hatted_data.label.str.endswith('*'))
        ]

        ax.errorbar(
            data_to_plot.value_mpcac_mass_hat,
            data_to_plot.value_ratio,
            xerr=data_to_plot.uncertainty_mpcac_mass_hat,
            yerr=data_to_plot.uncertainty_ratio,
            color=colour,
            marker=marker,
            label=f"$\\beta={beta}$",
            ls='none'
        )

    ax.legend(loc='best', frameon=False, ncol=2, columnspacing=0.5)

    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    fig.tight_layout(pad=0.28)
    fig.savefig(filename)
    plt.close(fig)