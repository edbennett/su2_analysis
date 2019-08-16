import matplotlib.pyplot as plt

from ..fitting import odr_fit
from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities

ENSEMBLES = [f'DB3M{number}' for number in range(1, 9)]


def fit_form_mPS_hat(consts, m_hat):
    # consts[0] = m_c_hat, consts[1] = B
    return (2 * consts[1] * (m_hat - consts[0])) ** 0.5


def generate(data, ensembles):
    filename = 'final_plots/fig6.pdf'
    set_plot_defaults()

    fig = plt.figure(figsize=(4.5, 5))
    ax = fig.subplots(nrows=2, sharex=True)
    f_ax = ax[0].twinx()

    ax[0].set_ylabel(r'$\hat{m}_{\mathrm{PS}}^2$')
    ax[1].set_ylabel(r'$\hat{m}_{\mathrm{PS}}^2 \hat{f}_{\mathrm{PS}}^2$')
    ax[1].set_xlabel(r'$\hat{m}_0 - \hat{m}_0^c$')
    f_ax.set_ylabel(r'$\hat{f}_{\mathrm{PS}}^2$')

    subset_data = data[data.label.isin(ENSEMBLES)]
    hatted_data = merge_and_hat_quantities(subset_data,
                                           ('g5_mass', 'g5_decay_const'))

    fit_result = odr_fit(
        fit_form_mPS_hat,
        hatted_data.value_m_hat, hatted_data.value_g5_mass_hat,
        hatted_data.uncertainty_m_hat, hatted_data.uncertainty_g5_mass_hat,
        (-2.0, 1.0)
    )
    value_m_c_hat = fit_result.beta[0]
    uncertainty_m_c_hat = fit_result.sd_beta[0]

    hatted_data['value_m_hat_diff'] = (hatted_data['value_m_hat']
                                       - value_m_c_hat)
    hatted_data['uncertainty_m_hat_diff'] = (
        hatted_data['uncertainty_m_hat'] ** 2 + uncertainty_m_c_hat ** 2
    ) ** 0.5

    hatted_data['value_mPSfPS_hat'] = (hatted_data.value_g5_mass_hat
                                       * hatted_data.value_g5_decay_const_hat)
    hatted_data['uncertainty_mPSfPS_hat'] = (
        hatted_data.value_g5_mass_hat ** 2
        * hatted_data.uncertainty_g5_decay_const_hat ** 2
        + hatted_data.value_g5_decay_const_hat ** 2
        * hatted_data.uncertainty_g5_mass_hat ** 2
    ) ** 0.5

    for plot_ax, quantity, marker, colour in (
            (ax[0], 'g5_mass', '+', 'green'),
            (f_ax, 'g5_decay_const', 'x', 'blue'),
            (ax[1], 'mPSfPS', '.', 'red')
    ):
        plot_ax.errorbar(
            hatted_data.value_m_hat_diff,
            hatted_data[f'value_{quantity}_hat'] ** 2,
            xerr=hatted_data.uncertainty_m_hat_diff,
            yerr=(2 * hatted_data[f'value_{quantity}_hat']
                  * hatted_data[f'uncertainty_{quantity}_hat']),
            fmt=marker,
            color=colour
        )

        plot_ax.set_xlim((0, None))
        plot_ax.set_ylim((0, None))

    plt.tight_layout()

    fig.savefig(filename)
    plt.close(fig)
