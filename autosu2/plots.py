from math import ceil

from matplotlib.pyplot import subplots, rc, close
from matplotlib.colors import XKCD_COLORS
from numpy import linspace, arange

from warnings import filterwarnings


COLOR_LIST = [XKCD_COLORS[f'xkcd:{colour}'] for colour in [
    'tomato red',
    'leafy green',
    'cerulean blue',
    'golden brown',
    'faded purple',
    'shocking pink',
    'pumpkin orange',
    'dusty teal',
    'red wine',
    'navy blue',
    'salmon'
]]

SYMBOL_LIST = '+.*o^x1v2'

filterwarnings("ignore", category=UserWarning, module="matplotlib")


def set_plot_defaults(fontsize=None, markersize=4, capsize=2, linewidth=1):
    if fontsize:
        font = {'size': fontsize}
    else:
        font = {}
    rc(
        'font',
        **{
            'family': 'serif',
            'serif': ['Computer Modern Roman'],
            **font
        }
    )
    rc('text', usetex=True)
    rc('lines', linewidth=linewidth, markersize=markersize)
    rc('errorbar', capsize=capsize)


def do_eff_mass_plot(masses, errors, filename=None, ymin=None, ymax=None,
                     tmin=None, tmax=None, m=None, m_error=None, ax=None,
                     colour='red', marker='s', label=None):
    assert (filename is not None) or (ax is not None)

    if not ax:
        fig, ax = subplots()
        local_ax = True
    else:
        local_ax = False

    if tmin is None:
        tmin = 0
    if tmax is None:
        tmax = len(masses) + 1
    t_range_start = ceil(tmin)
    t_range_end = min(len(masses), ceil(tmax))

    ax.errorbar(
        arange(t_range_start, t_range_end),
        masses[t_range_start:t_range_end],
        yerr=errors[t_range_start:t_range_end],
        fmt=marker,
        color=colour,
        label=label
    )

    if local_ax:
        ax.set_xlim((tmin, tmax))
        if ymin is None and ymax is None:
            ax.autoscale(axis='y', tight=True)
        else:
            ax.set_ylim((ymin, ymax))

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$m_{\mathrm{eff}}$')

    if m and m_error:
        if not tmin:
            tmin = 0
        if not tmax:
            tmax = len(masses)
        # ax.plot((tmin, tmax), (m, m), color=colour)
        ax.fill_between(
            # tmin + 1 due to mixing of adjacent points when calculating
            # effective mass
            (tmin + 1, tmax),
            (m + m_error, m + m_error),
            (m - m_error, m - m_error),
            facecolor=colour,
            alpha=0.4
        )

    if local_ax:
        fig.tight_layout()
        fig.savefig(filename)
        close(fig)


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
    close(fig)
