import matplotlib.pyplot as plt

from ..plots import set_plot_defaults, COLOR_LIST, SYMBOL_LIST
from ..derived_observables import merge_and_add_mhat2


def fit_form(consts, mhat2):
    # consts[0] = w0X, consts[1] = k1
    return consts[0] * (1 + consts[1] * mhat2)


def single_plot(all_data, ax, beta, colour, symbol):
    ax.set_ylabel(r'$a / w_0$')

    # Get data
    data_subset = all_data[all_data.beta == beta]
    merged_data = merge_and_add_mhat2(data_subset)

    # Plot data
    ax.errorbar(
        merged_data.value_mhat2,
        1 / merged_data.value_w0,
        xerr=merged_data.uncertainty_mhat2,
        yerr=merged_data.uncertainty_w0 / merged_data.value_w0 ** 2,
        fmt=symbol,
        color=colour,
        label=f'{beta}'
    )


def generate(data):
    filename = 'final_plots/fig3.pdf'
    set_plot_defaults()

    fig = plt.figure(figsize=(4, 2.8))
    ax = fig.subplots()

    ax.set_xlabel(r'$\hat{m}_{\mathrm{PS}}^2$')
    for beta, colour, symbol in zip((6.9, 7.05, 7.2, 7.4, 7.5),
                                    COLOR_LIST, SYMBOL_LIST):
        single_plot(data, ax, beta, colour, symbol)

    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    ax.legend(loc=0, frameon=False, columnspacing=0.6,
              ncol=5, handletextpad=0.1, bbox_to_anchor=(0.2, 0, 0.8, 0.15))
    ax.text(0.05, 0.04, r'$\beta$', transform=ax.transAxes)
    fig.tight_layout()

    fig.savefig(filename)
    plt.close(fig)
