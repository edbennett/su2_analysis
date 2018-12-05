from matplotlib.pyplot import subplots
from plots import set_plot_defaults
from data import read_db


def main():
    set_plot_defaults(fontsize=9, markersize=2)

    fig, ax = subplots(figsize=(3, 2))

    for style, beta in (('ro', '6.9'),
                        ('ms', '7.2'),
                        ('bD', '7.5')):
        data = read_db(f'processed_data/nf2_FUN/b{beta}.dat')
        ax.errorbar(
            (data.w0c * data.g5_mass) ** 2,
            1 / data.w0c,
            yerr=data.w0c_error / data.w0c ** 2,
            fmt=style
        )

    ax.set_xlim((0, 1.1))
    ax.set_ylim((0, 1.3))
    ax.set_xlabel(r'$(w_0\,m_{\mathrm{PS}})^2$')
    ax.set_ylabel(r'$a / w_0$')

    fig.tight_layout()
    fig.savefig('final_plots/fig2.pdf')


if __name__ == '__main__':
    main()
