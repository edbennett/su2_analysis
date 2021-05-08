#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import lsqfit
import gvar as gv

from ..derived_observables import merge_quantities
from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

from .common import beta_colour_marker


def Xpt_fit_form(mpcac_w0, p):
    am, mhat, w0 = mpcac_w0['data']
    result = 2 * mhat * p['B'] * (1 + p['L'] * mhat + p['D1'] * mhat * np.log(p['D2'] * mhat)) + p['W1'] * am + p['W2'] / w0 ** 2
    return {'data': result}


def Xpt_fit(hatted_data):
    mpcac_dict = {
        'data': (hatted_data['value_mpcac_mass'].values,
                 hatted_data['value_mpcac_mass_hat'].values,
                 hatted_data['value_w0'].values)
    }

    mg5_dict = {
        'data': gv.gvar(
            hatted_data['value_g5_mass_hat_squared'].values,
            np.diag(hatted_data['uncertainty_g5_mass_hat_squared'].values) ** 2
        )
    }

    priors = {
        'B': gv.gvar(1, 20),
        'L': gv.gvar(1, 20),
        'D1': gv.gvar(1, 20),
        'log(D2)': gv.gvar(1, 20),
        'W1': gv.gvar(1, 20),
        'W2': gv.gvar(1, 20)
    }

    return lsqfit.nonlinear_fit(
        data=(mpcac_dict, mg5_dict), prior=priors, fcn=Xpt_fit_form, debug=True
    )


def generate(data, ensembles):
    filename = 'final_plots/Xpt.pdf'

    set_plot_defaults(markersize=3, capsize=1, linewidth=0.5)

    observables = 'g5_mass', 'gk_mass'
    extra_observables = ('mpcac_mass', 'g5_decay_const')
    observable_labels = r'\gamma_5', r'\gamma_k'

    merged_data = merge_quantities(
        data, observables + extra_observables
    ).dropna(subset=('value_mpcac_mass',))

    hatted_data = merge_and_hat_quantities(
        data,
        ('mpcac_mass', 'g5_mass',)
    ).dropna(subset=('value_mpcac_mass',))

    fit_result = Xpt_fit(hatted_data)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for beta, colour, marker in beta_colour_marker:
        subset_data = hatted_data[hatted_data.beta == beta]
        subset_mpcac_dict = {
            'data': (subset_data['value_mpcac_mass'].values,
                     subset_data['value_mpcac_mass_hat'].values,
                     subset_data['value_w0'].values)
        }
        ax.errorbar(
            subset_data.value_mpcac_mass_hat,
            subset_data.value_g5_mass_hat_squared,
            xerr=subset_data.uncertainty_mpcac_mass_hat,
            yerr=subset_data.uncertainty_g5_mass_hat_squared,
            fmt='.',
            label=f'$\\beta={beta}$',
            color=colour,
            marker=marker
        )
        ax.scatter(
            subset_data.value_mpcac_mass_hat,
            gv.mean(Xpt_fit_form(subset_mpcac_dict, fit_result.p)['data']),
            marker='o',
            facecolor='none',
            color=colour,
            s=16,
            linewidths=0.5
        )

    ax.scatter(
        [np.nan], [np.nan],
        marker='o', facecolor='none', color='darkgray', s=16, linewidths=0.5,
        label=r'$\chi$PT result'
    )

    ax.set_ylim((0, None))
    ax.set_xlim((0, None))
    ax.set_ylabel(r'$w_0^2 M_{\gamma_5}^2$')
    ax.set_xlabel(r'$w_0 m_{\mathrm{PCAC}}$')
    ax.legend(loc='upper left', frameon=False, handletextpad=0, borderaxespad=0.2)

    fig.tight_layout(pad=0.25)
    fig.savefig(filename)
