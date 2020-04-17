import matplotlib.pyplot as plt
from numpy import linspace
from scipy.optimize import curve_fit

from ..plots import set_plot_defaults


def fit_form_mPS(m, B, m_c, D):
    return (2 * B * (m - m_c)) ** (1 / (1 + D))


def generate(data, ensembles):
    set_plot_defaults()
    for beta, m_index_max in ((2.05, 9), (2.1, 6), (2.15, 7), (2.2, 11)):
        filename = f'auxiliary_plots/crit_mf_b{beta}.pdf'

        fig, ax = plt.subplots(figsize=(4.5, 3))
        ax.set_xlabel(r'$m$')
        ax.set_ylabel(r'$m_{\pi}$')

        data_to_plot = data[(data.beta == beta) &
                            (data.observable == 'g5_mass') &
                            ~(data.label.str.endswith('*'))]
        ax.errorbar(data_to_plot.m, data_to_plot.value,
                    yerr=data_to_plot.uncertainty,
                    fmt='.')

        data_to_fit = data_to_plot.sort_values(by='value').iloc[:5]
        (B, m_c, D), pcov = curve_fit(
            fit_form_mPS,
            data_to_fit.m, data_to_fit.value,
            p0=(1.0, -2.0, 1.0), method='trf',
            sigma=data_to_fit.uncertainty, absolute_sigma=True
        )
        breakpoint()
        m_max = data_to_plot.m.max()
        m_range = linspace(m_c, m_max, 1000)
        ax.plot(m_range, fit_form_mPS(m_range, B, m_c, D))

        for L in (12, 16, 24, 32):
            ax.plot((m_c, m_max), (10.8 / L, 10.8 / L), label=f'$10.8 / {L}$')

        ax.legend(loc='lower right', frameon=False, ncol=2, columnspacing=1.0)
        ax.set_title(f'$\\beta={beta}: B={B:.2f}, m_c={m_c:.4f}, D={D:.2f}$')
        fig.tight_layout()
        fig.savefig(filename)
        plt.close(fig)
