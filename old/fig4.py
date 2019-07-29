from matplotlib.pyplot import subplots
from numpy import inf, linspace, logical_and, isnan
from scipy.optimize import curve_fit
from plots import set_plot_defaults
from data import read_db


def fit_form(x, a, b, c):
    return c * (x - a) ** 2 + b * (x - a)


def plot_single_beta(x_points, y_points, y_error, beta, xlabel, filename,
                     color=None, fit_form=None, fit_params=None):
    fig, ax = subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$(w_0m_{\mathrm{PS}}^2)^2'
                  r'(Z_{\mathrm{av}}w_0f_{\mathrm{PS}})^2$')
    ax.errorbar(
        x_points,
        y_points,
        yerr=y_error,
        color=color,
        fmt='o',
        label='Data'
    )
    ax.set_ylim([0, None])

    if fit_form:
        x_range = linspace(0, max(x_points), 1000)
        ax.plot(x_range, fit_form(x_range, *fit_params), label='Fit')
        fig.legend(loc=0)

    fig.tight_layout()
    fig.savefig(filename)


def fit_plot_single_beta(ax, beta, color, fmt, wm_min=-inf, wm_max=inf):
    data = read_db(f'processed_data/nf2_FUN/b{beta}.dat')
    x_points = (data.w0c * data.bare_mass).values
    y_points = (data.w0c ** 2 * data.g5_mass *
                data.Zav * data.g5_decay_const) ** 2
    y_error = y_points * (
        4 * (data.w0c_error / data.w0c) ** 2 +
        2 * (data.g5_mass_error / data.g5_mass) ** 2 +
        2 * (data.Zav_error / data.Zav) ** 2 +
        2 * (data.g5_decay_const_error / data.g5_decay_const) ** 2
    ) ** 0.5

    plot_single_beta(x_points, y_points, y_error, beta, r'$w_0m_0$',
                     f'processed_data/nf2_FUN/gmor_b{beta}.pdf', color)

    points_to_use = logical_and(x_points > wm_min,
                                x_points < wm_max,
                                ~isnan(x_points))
    print(x_points, wm_min, wm_max)
    print(points_to_use)
    fit_result, fit_covariance = curve_fit(
        fit_form,
        x_points[points_to_use],
        y_points[points_to_use],
        sigma=y_error[points_to_use],
        absolute_sigma=True,
    )
    print(f"w0 m0 = {fit_result[0]} ± {fit_covariance[0][0] ** 0.5}")
    print(f"B = {fit_result[1]} ± {fit_covariance[1][1] ** 0.5}")
    print(f"C = {fit_result[2]} ± {fit_covariance[2][2] ** 0.5}")

    plot_single_beta(x_points - fit_result[0], y_points, y_error, beta,
                     r'$w_0(m_0 - m_0^*)$',
                     f'processed_data/nf2_FUN/gmor_fit_b{beta}.pdf', color,
                     fit_form=fit_form, fit_params=fit_result)

    ax.errorbar(
        x_points - fit_result[0],
        data.w0c ** 2 * (
            data.g5_decay_const ** 2 +
            data.gk_decay_const ** 2 +
            data.g5gk_decay_const ** 2
        ),
        yerr=2 * data.w0c ** 2 * (
            data.g5_decay_const * data.g5_decay_const_error +
            data.gk_decay_const * data.gk_decay_const_error +
            data.g5gk_decay_const * data.g5gk_decay_const_error
        ),
        color=color,
        fmt=fmt,
        label=(r'$\beta=' f'{beta}$')
    )


def main():
    set_plot_defaults(fontsize=8, markersize=2)
    fig, ax = subplots(figsize=(3, 2))

    ax.set_xlabel(r'$w_0(m_0 - m_0^*)$')
    ax.set_ylabel(r'$(w_0 f_0)^2$')

#    fit_plot_single_beta(ax, '6.9', 'red', fmt='o', wm_min=-1.06, wm_max=-0.72)
    fit_plot_single_beta(ax, '7.2', 'blue', fmt='s')

    ax.legend(loc=0)

    fig.tight_layout()
    fig.savefig('final_plots/fig4.pdf')


if __name__ == '__main__':
    main()
