import matplotlib.pyplot as plt
from numpy import linspace, asarray
from uncertainties import ufloat

from ..fitting import odr_fit, confpred_band
from ..plots import set_plot_defaults
from ..tables import generate_table_from_content
from ..derived_observables import merge_and_add_mhat2


def fit_form(consts, mhat2):
    # consts[0] = w0X, consts[1] = k1
    return consts[0] * (1 + consts[1] * mhat2)


def single_plot_and_fit(all_data, ax, beta, fit_max, ylabel=None):
    ax.set_xlabel(r'$\hat{m}_{\mathrm{PS}}^2$')
    if ylabel:
        ax.set_ylabel(ylabel)

    # Get data
    data_subset = all_data[all_data.beta == beta]
    merged_data = merge_and_add_mhat2(data_subset)

    # Split data into data to be fitted and data to be plotted only
    fit_data = merged_data[merged_data.value_mhat2 <= fit_max]
    unfit_data = merged_data[merged_data.value_mhat2 > fit_max]

    # Plot data
    for data, colour in (fit_data, 'blue'), (unfit_data, 'red'):
        ax.errorbar(
            data.value_mhat2,
            data.value_w0,
            xerr=data.uncertainty_mhat2,
            yerr=data.uncertainty_w0,
            fmt='+',
            color=colour
        )
    ax.set_xlim((0, None))

    # Fit data
    fit_result = odr_fit(
        fit_form, fit_data.value_mhat2, fit_data.value_w0,
        fit_data.uncertainty_mhat2, fit_data.uncertainty_w0,
        (1, -0.1)
    )

    w0X, f1 = fit_result.beta

    # Plot fit results
    mhat2_range = linspace(*ax.get_xlim(), 1000)
    dfdp = asarray([1 + f1 * mhat2_range, w0X * mhat2_range])

    fit_line, fit_lower, fit_upper = confpred_band(
        mhat2_range,
        dfdp,
        fit_result,
        fit_form,
        err=fit_result.sd_beta[0],
        abswei=True
    )
    # fit_lower = fit_form((w0X - w0X_err, k1 - k1_err), mhat2_range)
    # fit_upper = fit_form((w0X + w0X_err, k1 + k1_err), mhat2_range)
    ax.fill_between(mhat2_range, fit_lower, fit_upper,
                    color='blue', alpha=0.2, lw=0)

    return fit_result


def table2(values):
    filename = 'table2.tex'
    columns = [r'$\beta$', None, r'$w_0^\chi / a$', r'$\tilde{k}_1$', None,
               r'$\chi^2/N_{\mathrm{d.o.f.}}$']
    table_content = []

    for beta, fit in values:
        w0X = ufloat(fit.beta[0], fit.sd_beta[0])
        k1 = ufloat(fit.beta[1], fit.sd_beta[1])
        chi2 = fit.res_var
        table_content.append(
            f'    {beta} & {w0X:.1ufS} & {k1:2ufS} & {chi2:.1}'
        )

    generate_table_from_content(filename, table_content, columns=columns)


def generate(data):
    filename = 'final_plots/fig2.pdf'
    set_plot_defaults()

    fig = plt.figure(figsize=(6, 2))
    ax = fig.subplots(ncols=2)

    fit_69 = single_plot_and_fit(data, ax[0], 6.9, 0.4, ylabel=r'$w_0 / a$')
    fit_72 = single_plot_and_fit(data, ax[1], 7.2, 0.45)
    table2(((6.9, fit_69), (7.2, fit_72)))

    plt.tight_layout()

    fig.savefig(filename)
    plt.close(fig)
