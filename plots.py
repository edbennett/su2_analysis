from matplotlib.pyplot import subplots, rcParams
from numpy import linspace


def set_plot_defaults():
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['lines.linewidth'] = 1
    rcParams['errorbar.capsize'] = 2
    rcParams['lines.markersize'] = 4


def do_eff_mass_plot(masses, errors, filename, ymin=None, ymax=None,
                     tmin=None, tmax=None, m=None, m_error=None):
    fig, ax = subplots()
    ax.errorbar(
        list(range(1, len(masses) + 1)),
        masses,
        yerr=errors,
        fmt='s'
    )
    ax.set_xlim((0, len(masses) + 1))
    ax.set_ylim((ymin, ymax))

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$m_{\mathrm{eff}}$')

    if m and m_error:
        if not tmin:
            tmin = 0
        if not tmax:
            tmax = len(masses)
        ax.plot((tmin, tmax), (m, m), color='red')
        ax.fill_between(
            (tmin, tmax),
            (m + m_error, m + m_error),
            (m - m_error, m - m_error),
            facecolor='red',
            alpha=0.4
        )
    fig.tight_layout()
    fig.savefig(filename)


def do_correlator_plot(correlator, errors, filename, channel_latex,
                       fit_function=None, fit_params=None, fit_legend='',
                       t_lowerbound=None, t_upperbound=None,
                       corr_lowerbound=None, corr_upperbound=None):
    if not t_lowerbound:
        t_lowerbound = 0
    if not t_upperbound:
        t_upperbound = len(correlator) - 1

    fig, ax = subplots()
    ax.errorbar(
        range(len(correlator)),
        correlator,
        yerr=errors,
        fmt='o',
        label='Data'
    )
    ax.set_xlim((t_lowerbound, t_upperbound))
    ax.set_ylim((corr_lowerbound, corr_upperbound))
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$C_{' f'{channel_latex}' r'}(t)$')

    if fit_function:
        if not fit_params:
            fit_params = []
        t_range = linspace(t_lowerbound, t_upperbound, 1000)
        ax.plot(
            t_range,
            fit_function(t_range, *fit_params),
            label=fit_legend
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(filename)
