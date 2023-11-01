import lsqfit
import gvar as gv
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray, linspace, ravel
from uncertainties import ufloat

from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, table_row
from ..derived_observables import merge_quantities


def quadratic(x, a, b, c):
    '''
    A simple quadratic, to keep the fit consistent with the plot
    '''
    return a + b * x + c * x ** 2


def fit(merged_data, betas_to_fit, gamma_star=None):
    L = merged_data.L.values
    beta_values = merged_data.beta.values
    mpcac = merged_data.value_mpcac_mass.values
    mpcac_e = merged_data.uncertainty_mpcac_mass.values
    mg5 = merged_data.value_g5_mass.values
    mg5_e = merged_data.uncertainty_g5_mass.values * 2

    def fit_form(all_mpcac, p):
        LMH = {}
        if gamma_star is None:
            ym = p['ym']
        else:
            ym = 1 + gamma_star

        for beta, (L, mass) in all_mpcac.items():
            x = L * mass ** (1 / ym)
            LMH[beta] = (
                quadratic(x, p['alpha0'], p['alpha1'], p['alpha2'])
                * (1
                   + p['c'][betas_to_fit.index(beta)]
                   * mass ** (-p['y0'] / ym))
            )
        return LMH

    mpcac_dict = {
        beta: (L[beta_values == beta], mpcac[beta_values == beta])
        for beta in betas_to_fit
    }

    mg5_dict = {
        beta: (
            L[beta_values == beta]
            * gv.gvar(
                mg5[beta_values == beta],
                np.diag(mg5_e[beta_values == beta]) ** 2
            )
        )
        for beta in betas_to_fit
    }

    priors = {
        'alpha0': gv.gvar(0.1, 5),
        'alpha1': gv.gvar(0.1, 5),
        'alpha2': gv.gvar(0.1, 5),
        'c': [gv.gvar(-0.1, 5) for _ in betas_to_fit],
        'y0': gv.gvar(-2, 2),
    }
    if gamma_star is None:
        priors['ym'] = gv.gvar(1.7, 1)

    fit_result = lsqfit.nonlinear_fit(
        data=(mpcac_dict, mg5_dict), prior=priors, fcn=fit_form, debug=True
    )
    if gamma_star is not None:
        fit_result.p['ym'] = gamma_star + 1

    return fit_result


def plot(data, fit_result, betas_to_fit, filename=None):
    set_plot_defaults()
    fig, ax = plt.subplots()
    if type(fit_result.p['ym']) == float:
        ym = fit_result.p['ym']
        ym_sd = 0
    else:
        ym = fit_result.p['ym'].mean
        ym_sd = fit_result.p['ym'].sdev
    y0 = fit_result.p['y0'].mean
    omega = -fit_result.p['y0'] / fit_result.p['ym']
    c = [c0.mean for c0 in fit_result.p['c']]

    alpha0 = fit_result.p['alpha0'].mean
    alpha1 = fit_result.p['alpha1'].mean
    alpha2 = fit_result.p['alpha2'].mean

    for beta_index, beta in enumerate(betas_to_fit):
        subset_data = data[data.beta == beta]
        L = subset_data.L

        ax.errorbar(
            L * subset_data.value_mpcac_mass ** (1 / ym),
            L * subset_data.value_g5_mass / (
                1 + c[beta_index] * subset_data.value_mpcac_mass ** (-y0 / ym)
            ),
            yerr=L * subset_data.uncertainty_g5_mass / (
                1 + c[beta_index] * subset_data.value_mpcac_mass ** (-y0 / ym)
            ),
            fmt='.', label=r'$\beta=' f'{beta}$'
        )

    ax.set_xlabel(r'$Lm_{\mathrm{PCAC}}^{1/y_m}$')
    ax.set_ylabel(r'$LM_{\gamma_5}/(1+c_0m^\omega)$')
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    xlim = ax.get_xlim()
    xrange = linspace(*xlim, 1000)
    ax.plot(
        xrange, quadratic(xrange, alpha0, alpha1, alpha2),
        label=f'${alpha0:.2}+{alpha1:.2}x+{alpha2:.2}x^2$'
    )

    ax.set_title(
        r'$\gamma_*=' f'{ufloat(ym, ym_sd) - 1:.2uS}$, '
                 r'$\omega=' f'{ufloat(omega.mean, omega.sdev):.2uS}$'
    )

    ax.legend(frameon=False)


    if filename is None:
        plt.show()
    else:
        if filename != '':
            fig.savefig(filename)
        plt.close(fig)


def tabulate(fit_results, filename):
    columns = [r'$\beta_{\mathrm{min}}$', None,
               r'$\gamma_*$', r'$\omega$', None,
               r'$y_{\mathrm{m}}$', r'$y_0$', None,
               r'$\alpha_0$', r'$\alpha_1$', r'$\alpha_2$', None,
               r'$c_{\beta}$', None,
               r'$\chi^2/\mathrm{d.o.f.}$']
    table_content = []
    add_hline = False
    num_format = r'${}$' #r'${:.2uSL}$'
    for betas, fit_result in fit_results:
        if betas is None:
            add_hline = True
            continue

        p = fit_result.p
        ym = p['ym']
        gamma_star = ym - 1
        if type(gamma_star) == float:
            gamma_star_text = f'{gamma_star:.2f}'
            ym_text = f'{ym:.2f}'
        else:
            gamma_star_text = f'{gamma_star}'
            ym_text = f'{ym}'

        omega = -p['y0'] / p['ym']

        row_content = (
            [num_format.format(value)
             for value in (min(betas), gamma_star_text, omega, ym_text,
                           p['y0'], p['alpha0'], p['alpha1'], p['alpha2'])]
            + ['$' + ', '.join(map(str, p['c'])) + '$',
               f'{fit_result.chi2 / fit_result.dof:.2f}']
        )
        table_content.append(
            ('    \\hline\n' if add_hline else '')
            + table_row(row_content)
        )
        add_hline = False

    generate_table_from_content(filename, table_content, columns=columns)


def generate(data, ensembles):
    filename = 'auxiliary_plots/nearmarginal_{}beta{}.pdf'
    table_filename = 'nearmarginal{}.tex'
    fit_results_free = []
    fit_results_constrained = []

    betas_to_fit_set = [
        (2.1, 2.15, 2.2, 2.3, 2.4),
        (2.15, 2.2, 2.3, 2.4),
        (2.2, 2.3, 2.4),
        (2.3, 2.4),
    ]

    merged_data = merge_quantities(
        data[data.Nf == 1], ['g5_mass', 'mpcac_mass']
    ).dropna(
        subset=('value_mpcac_mass', 'value_g5_mass')
    )

    for betas_to_fit in betas_to_fit_set:
        fit_result = fit(merged_data, betas_to_fit)
        fit_results_free.append((betas_to_fit, fit_result))
        plot(merged_data, fit_result, betas_to_fit,
             filename.format(len(betas_to_fit), ''))

        for gamma_star in linspace(0.1, 1.05, 20):
            fit_result = fit(merged_data, betas_to_fit, gamma_star=gamma_star)
            fit_results_constrained.append((betas_to_fit, fit_result))
            plot(merged_data, fit_result, betas_to_fit,
                 filename.format(len(betas_to_fit), f'_gamma{gamma_star:.2f}'))
        fit_results_constrained.append((None, None))

    tabulate(fit_results_free, table_filename.format(''))
    tabulate(fit_results_constrained[:-1], table_filename.format('_fixbeta'))
