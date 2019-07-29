from matplotlib.pyplot import subplots
from numpy import nan
from plots import set_plot_defaults
from data import read_db


def generate_legend(ax):
    ax[0].plot([nan], [nan], 'kd', label='Axial vector')
    ax[0].plot([nan], [nan], 'ks', label='Vector')
    ax[1].plot([nan], [nan], 'ko', label='Pseudoscalar')
    ax[1].plot([nan], [nan], 'kd', label='Axial vector')
    ax[1].plot([nan], [nan], 'ks', label='Vector')
    for axis in ax:
        axis.legend(loc=0, frameon=False)


def plot_single_beta(ax, beta, color):
    data = read_db(f'processed_data/nf2_FUN/b{beta}.dat')
    x_points = (data.w0c * data.g5_mass) ** 2
    ax[0].errorbar(
        x_points,
        data.w0c * data.g5gk_mass,
        yerr=(data.w0c_error * data.g5gk_mass +
              data.w0c * data.g5gk_mass_error),
        color=color,
        fmt='d'
    )
    ax[0].errorbar(
        x_points,
        data.w0c * data.gk_mass,
        yerr=(data.w0c_error * data.gk_mass +
              data.w0c * data.gk_mass_error),
        color=color,
        fmt='s'
    )

    ax[1].errorbar(
        x_points,
        data.w0c * data.g5_decay_const,
        yerr=(data.w0c_error * data.g5_decay_const +
              data.w0c * data.g5_decay_const_error),
        color=color,
        fmt='o'
    )
    ax[1].errorbar(
        x_points,
        data.w0c * data.g5gk_decay_const,
        yerr=(data.w0c_error * data.g5gk_decay_const +
              data.w0c * data.g5gk_decay_const_error),
        color=color,
        fmt='d'
    )
    ax[1].errorbar(
        x_points,
        data.w0c * data.gk_decay_const,
        yerr=(data.w0c_error * data.gk_decay_const +
              data.w0c * data.gk_decay_const_error),
        color=color,
        fmt='s'
    )

    ax[0].set_ylim((0.5, 1.5))
    ax[1].set_ylim((0, 0.27))
    for axis in ax:
        axis.set_xlim((0, 0.9))


def main():
    set_plot_defaults(fontsize=8, markersize=2)
    fig, ax = subplots(ncols=2, figsize=(6, 2))
    
    ax[0].set_xlabel(r'$(w_0 m_{\mathrm{PS}})^2$')
    ax[1].set_xlabel(r'$(w_0 m_{\mathrm{PS}})^2$')

    ax[0].set_ylabel(r'$w_0 m_M$')
    ax[1].set_ylabel(r'$w_0 f_M$')

    plot_single_beta(ax, '6.9', 'red')
    plot_single_beta(ax, '7.2', 'blue')

    generate_legend(ax)

    fig.tight_layout()
    fig.savefig('final_plots/fig3.pdf')


if __name__ == '__main__':
    main()
