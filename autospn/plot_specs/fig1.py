import matplotlib.pyplot as plt
from matplotlib.colors import XKCD_COLORS

from numpy import nan

from ..plots import set_plot_defaults

ENSEMBLES = (
    'DB3M1', 'DB3M2', 'DB3M3', 'DB3M4', 'DB3M5', 'DB3M6', 'DB3M7', 'DB3M8',
)
XS = (0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0)
ERROR_DIGITS = 2
EXPONENTIAL = False

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


def generate(data):
    filename = 'final_plots/fig1.pdf'

    set_plot_defaults()

    fig = plt.figure(figsize=(6, 3.5))
    t_ax = plt.subplot2grid((1, 5), (0, 0), rowspan=1, colspan=2)
    w_ax = plt.subplot2grid((1, 5), (0, 2), rowspan=1, colspan=2)
    key_ax = plt.subplot2grid((1, 5), (0, 4), rowspan=1, colspan=1)
    key_ax.set_axis_off()

    t_ax.set_xlabel(r'$am_0$')
    w_ax.set_xlabel(r'$am_0$')
    t_ax.set_ylabel(r'$\sqrt{8t_0}/a$')
    w_ax.set_ylabel(r'$w_0/a$')

    for label, symbol in (('Plaquette', '+'), ('Clover', '.')):
        key_ax.plot([nan], [nan], symbol, label=label, color='black')

    for X, colour in zip(XS, COLOR_LIST):
        for ax, observable in ((t_ax, 's8t0'), (w_ax, 'w0')):
            for stencil, symbol in (('p', '+'), ('c', '.')):
                data_to_plot = data[
                    data.label.isin(ENSEMBLES) &
                    (data.observable == f'{observable}{stencil}') &
                    (data.free_parameter == X)
                ]
                ax.errorbar(data_to_plot.m, data_to_plot.value,
                            fmt=symbol,
                            yerr=data_to_plot.uncertainty,
                            color=colour)
        key_ax.errorbar(
            [nan], [nan], yerr=nan, fmt='.', markersize=0,
            label=r'${:.02}$'.format(X), color=colour, capsize=3
        )

    key_ax.legend(loc=0, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(right=1)
    plt.savefig(filename)
    plt.close(fig)
