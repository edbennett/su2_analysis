from matplotlib.pyplot import subplots
from scipy.optimize import curve_fit
from numpy import exp, linspace
from plots import set_plot_defaults
from data import read_db


def fit_form(mPS_L, A, B):
    return B + A * exp(-B * mPS_L)


def fit_plot_single(axis, data, xmin, xmax, ymin, ymax):
    (A, B), _ = curve_fit(
        fit_form,
        data.L * data.g5_mass,
        data.g5_mass,
        sigma=data.g5_mass_error,
        absolute_sigma=True
    )
    print(A, B, _)
    axis.errorbar(
        data.L * data.g5_mass,
        data.g5_mass,
        yerr=data.g5_mass_error,
        fmt='o',
        color='red'
    )
    plot_range = linspace(xmin, xmax, 1000)
    axis.plot(
        plot_range,
        fit_form(plot_range, A, B),
        'b--'
    )
    axis.set_xlim((xmin, xmax))
    axis.set_ylim((ymin, ymax))
    axis.set_xlabel(r'$m_{\mathrm{PS}}L$')
    axis.set_ylabel(r'$a\ m_{\mathrm{PS}}$')


def main():
    set_plot_defaults(fontsize=9)
    data = read_db('processed_data/nf2_FUN/b7.2.dat')
    left_data = data[data.bare_mass == -0.77]
    right_data = data[data.bare_mass == -0.79]

    fig, ax = subplots(ncols=2, figsize=(6, 2))
    fit_plot_single(ax[0], left_data, 5.75, 11.25, 0.42, 0.437)
    fit_plot_single(ax[1], right_data, 4.25, 8.75, 0.31, 0.35)

    fig.tight_layout()
    fig.savefig('final_plots/fig1.pdf')


if __name__ == '__main__':
    main()
