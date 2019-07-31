import matplotlib.pyplot as plt
from numpy import nan, exp, linspace

from ..plots import set_plot_defaults
from ..tables import generate_table_from_db
from ..fitting import odr_fit

ENSEMBLES = (
    ('DB3M4*', 'DB3M4**', 'DB3M4'),
    ('DB3M6*', 'DB3M6**', 'DB3M6')
)

ERROR_DIGITS = 2
EXPONENTIAL = False


def table3(data, m_PS_inf_values):
    filename = 'table3.tex'
    columns = ['Ensemble', None, r'$am_0$', r'$N_t \times N_s^3', None,
               r'$am_{\mathrm{PS}}$', r'am_{\mathrm{V}}$', None,
               r'$m_{\mathrm{PS}}^{\mathrm{inf.}}L$']
    constants = ['m', 'V']
    multirow = {'V': False, 'm': True}
    observables = ['g5_mass', 'gk_mass', 'LmPSinf']
    ensembles = []
    for (ensemble_set,
         (m_PS_inf, m_PS_inf_err)) in zip(ENSEMBLES, m_PS_inf_values):
        for ensemble in ensemble_set:
            ensembles.append(ensemble)
            datum = data[(data.label == ensemble) &
                         (data.observable == 'g5_mass')].copy()
            datum.observable = 'LmPSinf'
            datum.value = datum.L * m_PS_inf
            datum.uncertainty = datum.L * m_PS_inf_err
            data = data.append(datum)
        ensembles.append(None)
    ensembles = ensembles[:-1]

    generate_table_from_db(
        data=data,
        ensembles=ensembles,
        observables=observables,
        filename=filename,
        columns=columns,
        constants=constants,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL,
        multirow=multirow
    )


def fit_form(consts, L):
    # consts[0] = A_M, consts[1] = m_PS^inf
    return consts[1] * (
        1 + consts[0] * exp(-consts[0] * L) / (consts[0] * L) ** (3/2)
    )


def single_plot_and_fit(all_data, ax, ensembles,
                        xlim=(None, None), ylabel=None):
    ax.set_xlabel(r'$m_{\mathrm{PS}}^{\mathrm{inf.}}L$')
    if ylabel:
        ax.set_ylabel(ylabel)

    # Get data
    data_subset = all_data[all_data.label.isin(ensembles) &
                           (all_data.observable == 'g5_mass')]
    assert len(data_subset) <= len(ensembles)

    # Fit data
    fit_result = odr_fit(
        fit_form, data_subset.L, data_subset.value,
        yerr=data_subset.uncertainty,
        p0=(1, 0.4)
    )

    A_M, m_PS_inf = fit_result.beta

    # Plot data
    ax.errorbar(
        data_subset.L * m_PS_inf,
        data_subset.value,
        yerr=data_subset.uncertainty,
        fmt='+',
        color='blue'
    )
    ax.set_xlim(xlim)

    # Plot fit results
    L_m_PS_inf_range = linspace(*ax.get_xlim(), 1000)
    ax.plot(L_m_PS_inf_range, fit_form(fit_result.beta, L_m_PS_inf_range),
            'b--')
    ax.plot(ax.get_xlim(), (m_PS_inf, m_PS_inf), 'k-')

    return fit_result


def generate(data):
    filename = 'final_plots/fig5.pdf'

    set_plot_defaults()

    fig = plt.figure(figsize=(6, 2.2))
    ax = fig.subplots(ncols=2)

    m4_result = single_plot_and_fit(
        data, ax[0], ENSEMBLES[0],
        xlim=(5.5, 11.5), ylabel=r'$a\,m_{\mathrm{PS}}$'
    )
    m6_result = single_plot_and_fit(
        data, ax[1], ENSEMBLES[1],
        xlim=(4, 9.5)
    )
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    table3(data, (
        (m4_result.beta[1], m4_result.sd_beta[1]),
        (m6_result.beta[1], m6_result.sd_beta[1])
    ))
