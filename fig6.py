from matplotlib.pyplot import subplots
from scipy.optimize import curve_fit
from numpy import exp, linspace
from plots import set_plot_defaults
from data import read_db


def plot_single_beta(data, ax, xlim, xticks=None):
    ax.set_xlabel(r'$a\,m_0$')
    ax.set_xlim(xlim)
    if xticks:
        ax.set_xticks(xticks)
    ax.errorbar(
        data.bare_mass,
        data.hot_plaquette,
        yerr=data.hot_plaquette_error,
        color='red',
        fmt='.'
    )
    ax.errorbar(
        data.bare_mass,
        data.cold_plaquette,
        yerr=data.cold_plaquette_error,
        color='blue',
        fmt='+'
    )


def main():
    set_plot_defaults(fontsize=9)
    data = read_db('processed_data/nf3_ASY/all.dat')

    fig, ax = subplots(ncols=3, sharey=True, figsize=(6, 2))
    ax[0].set_ylabel(r'$\langle P\rangle$')
    ax[0].set_ylim((0.49, 0.59))
    plot_single_beta(data[data.beta == 6.4], ax[0], (-1.19, -1.09),
                     xticks=(-1.18, -1.14, -1.10))
    plot_single_beta(data[data.beta == 6.5], ax[1], (-1.115, -1.063),
                     xticks=(-1.11, -1.09, -1.07))
    plot_single_beta(data[data.beta == 6.6], ax[2], (-1.105, -0.985))

    fig.tight_layout()
    fig.savefig('final_plots/fig6.pdf')


if __name__ == '__main__':
    main()
