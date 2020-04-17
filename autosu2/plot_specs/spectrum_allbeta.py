import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_and_hat_quantities


def generate(data, ensembles):
    set_plot_defaults()

    use_pcac = True

    filename = f'auxiliary_plots/spectrum.pdf'
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(4.5, 5))
    hatted_data = merge_and_hat_quantities(
        data,
        ('mpcac_mass', 'g5_mass', 'g5_decay_const',
         'g5gk_mass', 'g5gk_decay_const', 'id_mass', 'id_decay_const')
    )
    if use_pcac:
        axes[1].set_xlabel(r'$w_0 m_{\mathrm{PCAC}}$')
    else:
        axes[1].set_xlabel(r'$w_0 (m - m_c)$')

    axes[0].set_ylabel(r'$w_0 M$')
    axes[1].set_ylabel(r'$w_0 f$')

    for beta, colour, m_c in (
            (2.05, 'r', -1.5256),
            (2.1, 'g', -1.4760),
            (2.15, 'b', -1.4289),
            (2.2, 'k', -1.3932)
    ):
        data_to_plot = hatted_data[
            (hatted_data.beta == beta) &
            ~(hatted_data.label.str.endswith('*'))
        ]
        if use_pcac:
            mhat = data_to_plot.value_mpcac_mass_hat
            mhat_err = data_to_plot.uncertainty_mpcac_mass_hat
        else:
            mw0 = (data_to_plot.m - m_c) * data_to_plot.value_w0
            mw0_err = (data_to_plot.m - m_c) * data_to_plot.uncertainty_w0

        for observable, ax in zip(('mass', 'decay_const'), axes):
            for channel, symbol in (('g5', '.'), ('g5gk', 'x'), ('id', '+')):
                suffix = f'{channel}_{observable}_hat'
                ax.errorbar(mhat,
                            data_to_plot[f'value_{suffix}'],
                            xerr=mhat_err,
                            yerr=data_to_plot[f'uncertainty_{suffix}'],
                            fmt=f'{colour}{symbol}',
                            label=f'{channel}, $\\beta={beta}$')

        #data_to_fit = data_to_plot.sort_values(by='value').iloc[:5]
        #(B, m_c, D), pcov = curve_fit(
        #    fit_form_mPS, 
        #    data_to_fit.m, data_to_fit.value,
        #    p0=(1.0, -2.0, 1.0), method='trf',
        #    sigma=data_to_fit.uncertainty, absolute_sigma=True
        #)

        #m_max = data_to_plot.m.max()
        #m_range = linspace(m_c, m_max, 1000)
        #ax.plot(m_range, fit_form_mPS(m_range, B, m_c, D))

        #for L in (12, 16, 24, 32):
        #    ax.plot((m_c, m_max), (11 / L, 11 / L), label=f'$11.0 / {L}$')

    axes[0].legend(loc='lower right', frameon=False, ncol=2, columnspacing=1.0)

    axes[0].set_ylim((0, None))
    axes[1].set_ylim((0, None))

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
