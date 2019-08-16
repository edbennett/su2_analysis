import matplotlib.pyplot as plt
from numpy import concatenate, nan

from ..plots import set_plot_defaults, COLOR_LIST, SYMBOL_LIST
from ..derived_observables import merge_and_hat_quantities

QUANTITY_NAMES = {'f': 'renormalised_decay_const',
                  'm': 'mass'}
CHANNEL_LABELS = {'PS': 'g5', 'V': 'gk', 'AV': 'g5gk',
                  'T': 'g0gk', 'AT': 'g0g5gk', 'S': 'id'}


def set_up_legend(ax):
    # Legend titles have to be above the legend, but we want it to the left
    # An ax.text element will throw out the tight_layout; the text has to
    # be part of the legend.
    # So use a dummy point that won't display to get the extra legend entry
    ax[0].plot([nan], [nan], 'w', label=r'$\beta$:')
    ax[0].legend(loc=(-0.1, 1.05),
                 frameon=False, ncol=6, columnspacing=0.65, handletextpad=0.1)


def set_up_x_axis(ax):
    ax[-1].set_xlim((0, 0.85))
    ax[-1].set_xlabel(r'$\hat{m}_{\mathrm{PS}}^2$')


def generate(data, ensembles):
    set_plot_defaults(markersize=2)

    fig = {}
    fig[7] = plt.figure(figsize=(4, 7.5))
    fig[8] = plt.figure(figsize=(4, 5))
    fig[9] = plt.figure(figsize=(4, 7.5))

    axes = {}
    for index, rows in ((7, 3), (8, 2), (9, 3)):
        axes[index] = fig[index].subplots(nrows=rows, sharex=True)
    for ax in axes.values():
        set_up_x_axis(ax)

    betas = 6.9, 7.05, 7.2, 7.4, 7.5

    quantities_channels = [
        ('f', 'PS'), ('f', 'V'), ('f', 'AV'),
        ('m', 'V'), ('m', 'T'),
        ('m', 'S'), ('m', 'AV'), ('m', 'AT')
    ]
    columns = ([f'{CHANNEL_LABELS[channel]}_{QUANTITY_NAMES[quantity]}'
               for quantity, channel in quantities_channels]
               + ['g5_mass'])
    merged_data = merge_and_hat_quantities(data, columns)

    axes_list = concatenate([axes[7], axes[8], axes[9]])
    for (quantity, channel), column, ax in zip(
            quantities_channels, columns, axes_list
    ):
        ax.set_ylabel(r'$\hat{' f'{quantity}' r'}_{\mathrm{' f'{channel}'
                      r'}}^2$')
        for beta, colour, symbol in zip(betas, COLOR_LIST, SYMBOL_LIST):
            to_plot = merged_data[merged_data.beta == beta]
            ax.errorbar(to_plot[f'value_g5_mass_hat_squared'],
                        to_plot[f'value_{column}_hat_squared'],
                        xerr=to_plot[f'uncertainty_g5_mass_hat_squared'],
                        yerr=to_plot[f'uncertainty_{column}_hat_squared'],
                        fmt=symbol, color=colour, label=f'{beta}')

    for ax in axes.values():
        set_up_legend(ax)

    # f_AV is flat, so looks messy; start axis from 0 to avoid this
    axes[7][2].set_ylim((0, None))

    for fig_index in 7, 8, 9:
        fig[fig_index].tight_layout()
        fig[fig_index].savefig(f'final_plots/fig{fig_index}.pdf')
        plt.close(fig[fig_index])


if __name__ == '__main__':
    generate(None, None)
