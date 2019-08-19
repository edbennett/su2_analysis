from matplotlib import pyplot as plt
from numpy import (
    asarray, linspace, zeros_like, max, min, ceil, log10, round, row_stack,
    isnan, inf
)
from uncertainties import ufloat

from ..plots import set_plot_defaults, COLOR_LIST, SYMBOL_LIST
from ..tables import HLINE, generate_table_from_content
from ..fitting import odr_fit, confpred_band
from ..derived_observables import merge_and_hat_quantities
from .fig7_8_9 import QUANTITY_NAMES, CHANNEL_LABELS

X_AXIS_LIMIT = (0, 0.62)


def fit_form(consts, m_PS2_and_w0):
    # q_and_w0 is a 2xN array:
    #  - m_PS2_and_w0[0] is \hat{m}_{PS}^2;
    #  - m_PS2_and_w0[1] is w0
    # q refers to either f or m
    q_hat_squared_chi, L_zero, W_zero = consts
    return (q_hat_squared_chi * (1 + L_zero * m_PS2_and_w0[0])
            + W_zero / m_PS2_and_w0[1])


def set_up_axis(fig, axis_index,
                observable, channel, offset,
                num_channels, num_rows, xlim=X_AXIS_LIMIT):
    axis = plt.subplot2grid(
        shape=(num_rows, 4),
        loc=(axis_index // 2, offset),
        colspan=2,
        fig=fig
    )
    axis.set_xlabel(r'$\hat{m}_{\mathrm{PS}}^2$')
    axis.set_ylabel(r'$\hat{' f'{observable}'
                    r'}^2_{\mathrm{' f'{channel}' r'}}$')
    axis.set_xlim(xlim)
    return axis


def do_single_fit(data, channel_full_name):
    '''
    Extracts values from `data` for `channel_full_name`, and fits them using
    scipy.odr using the `fit_form` defined above as a function of both
    the PS mass and w0.
    '''

    # First, remove any NaNs, which will break ODR
    valid_rows = (data['value_g5_mass_hat_squared'].notna()
                  & data['uncertainty_g5_mass_hat_squared'].notna()
                  & data[f'value_{channel_full_name}'].notna()
                  & data[f'uncertainty_{channel_full_name}'].notna()
                  & data['value_w0'].notna()
                  & data['uncertainty_w0'].notna())

    m_PS2_values = data['value_g5_mass_hat_squared'][valid_rows]
    q_values = data[f'value_{channel_full_name}'][valid_rows]
    m_PS2_errors = data['uncertainty_g5_mass_hat_squared'][valid_rows]
    q_errors = data[f'uncertainty_{channel_full_name}'][valid_rows]
    w0_values = data['value_w0'][valid_rows]
    w0_errors = data['uncertainty_w0'][valid_rows]

    m_PS2_and_w0_values = row_stack([m_PS2_values, w0_values])
    m_PS2_and_w0_errors = row_stack([m_PS2_errors, w0_errors])
    fit_result = odr_fit(fit_form, m_PS2_and_w0_values, q_values,
                         m_PS2_and_w0_errors, q_errors, num_params=3)
    return fit_result


def fit_and_plot_single_channel(data, observable, channel, axis):
    observable_name = QUANTITY_NAMES[observable]
    channel_label = CHANNEL_LABELS[channel]

    # Plot data
    betas = 6.9, 7.05, 7.2, 7.4, 7.5
    for beta, colour, symbol in zip(betas, COLOR_LIST, SYMBOL_LIST):
        to_plot = data[data.beta == beta]
        m_PS2_values = to_plot['value_g5_mass_hat_squared']
        q_values = to_plot[
            f'value_{channel_label}_{observable_name}_hat_squared'
        ]
        m_PS2_errors = to_plot['uncertainty_g5_mass_hat_squared']
        q_errors = to_plot[
            f'uncertainty_{channel_label}_{observable_name}_hat_squared'
        ]

        axis.errorbar(
            m_PS2_values, q_values,
            xerr=m_PS2_errors, yerr=q_errors,
            color=colour, fmt=symbol
        )

    # Fit data
    channel_full_name = f'{channel_label}_{observable_name}_hat_squared'
    fit_result = do_single_fit(data, channel_full_name)
    fit_result_minus_heaviest = do_single_fit(
        data[data[f'value_{channel_full_name}']
             < max(data[f'value_{channel_full_name}'])],
        channel_full_name
    )
    fit_result_minus_coarsest = do_single_fit(
        data[data[f'value_w0'] < max(data[f'value_w0'])],
        channel_full_name
    )

    # Plot fit result
    q_squared_chi, L_zero, W_zero = fit_result.beta
    m_PS2_range = linspace(*X_AXIS_LIMIT, 1000)
    dfdp = asarray([1 + L_zero * m_PS2_range,
                    q_squared_chi * m_PS2_range,
                    zeros_like(m_PS2_range)])
    both_parameters_range = asarray([m_PS2_range,
                                     zeros_like(m_PS2_range) + inf])
    fit_line, fit_lower, fit_upper = confpred_band(
        both_parameters_range,
        dfdp,
        fit_result,
        fit_form,
        err=fit_result.sd_beta[0],
        abswei=True
    )
    axis.plot(m_PS2_range, fit_line, 'k--')
    axis.fill_between(m_PS2_range, fit_lower, fit_upper,
                      color='black', alpha=0.2, lw=0)

    return fit_result, fit_result_minus_heaviest, fit_result_minus_coarsest


def fit_and_plot(data, observable, channels, filename, figsize=None):
    fig = plt.figure(figsize=figsize)
    axes = []
    fit_results = []

    num_rows = len(channels) // 2 + len(channels) % 2
    offsets = [0, 2] * num_rows
    if len(channels) % 2 == 1:
        del offsets[-1]
        offsets[-1] = 1

    for axis_index, (channel, offset) in enumerate(zip(channels, offsets)):
        if channel == 'PS':
            extra_params = {'xlim': (0, 0.42)}
        else:
            extra_params = {'xlim': X_AXIS_LIMIT}

        axis = set_up_axis(fig, axis_index, observable, channel, offset,
                           len(channels), num_rows, **extra_params)
        axes.append(axis)
        fit_results.append(
            fit_and_plot_single_channel(data, observable, channel, axis)
        )

    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    return fit_results


def tabulate(fit_results, observables, channels, filename):
    table_content = []
    for observable_fit_results, letter, channels in zip(
            fit_results, observables, channels
    ):
        table_content.append(
            r' & $\hat{' f'{letter}' r'}_{M}^{2,\chi}$'
            r' & $L_{' f'{letter}' r',M}^0$'
            r' & $W_{' f'{letter}' r',M}^0$'
            r' & $\chi^2 / N_{\mathrm{d.o.f.}}$'
        )
        line = HLINE
        for channel_fit_results, channel in zip(
                observable_fit_results, channels
        ):
            row_content = [channel]
            fit_result_betas = [fit_result.beta
                                for fit_result in channel_fit_results]
            values = channel_fit_results[0].beta
            errors = channel_fit_results[0].sd_beta
            systematics = (max(fit_result_betas, axis=0)
                           - min(fit_result_betas, axis=0))
            chi2 = channel_fit_results[0].res_var
            for value, error, systematic in zip(values, errors, systematics):
                if isnan(value):
                    row_content.append('---')
                    continue

                value_error = ufloat(value, error)
                if isnan(systematic):
                    systematic_formatted = '---'
                else:
                    systematic_formatted = int(round(
                        systematic / (10 ** ceil(log10(systematic) - 2))
                    ))
                row_content.append(f'{value_error:.2ufS}'
                                   f'({systematic_formatted})')
            row_content.append(f'{chi2:.1f}')

            table_content.append(line + ' & '.join(row_content))
            line = '    '

        table_content[-1] += HLINE + HLINE
    generate_table_from_content(filename, table_content, table_spec='c|ccc|c')


def hat_and_filter(data, channels, observables):
    fPS_ensembles = (
        [f'DB1M{N}' for N in (5, 6, 7)]
        + [f'DB2M{N}' for N in (1, 2, 3)]
        + [f'DB3M{N}' for N in {5, 6, 7, 8}]
        + ['DB4M2']
    )
    data_filtered_fPS = data[
        data.observable.ne('g5_renormalised_decay_const')
        | data.label.isin(fPS_ensembles)
    ]

    columns_to_hat = [
        f'{CHANNEL_LABELS[channel]}_{QUANTITY_NAMES[quantity]}'
        for quantity, channel_group in zip(observables, channels)
        for channel in channel_group
    ] + ['g5_mass']

    hatted_data_filtered_fPS = merge_and_hat_quantities(data_filtered_fPS,
                                                        columns_to_hat)
    return hatted_data_filtered_fPS[
        hatted_data_filtered_fPS.value_g5_mass_hat_squared.le(0.6)
        | hatted_data_filtered_fPS.value_w0.ge(1.0)
    ]


def generate(data, ensembles):
    set_plot_defaults(linewidth=0.5, capsize=1)
    channels = (('PS', 'V', 'AV'), ('V', 'T', 'AV', 'AT', 'S'))
    observables = ('f', 'm')

    data_to_fit = hat_and_filter(data, channels, observables)
    fit_results_decay_const = fit_and_plot(
        data_to_fit,
        observable=observables[0],
        channels=channels[0],
        filename='final_plots/fig10.pdf',
        figsize=(6, 4)
    )
    fit_results_mass = fit_and_plot(
        data_to_fit,
        observable=observables[1],
        channels=channels[1],
        filename='final_plots/fig11.pdf',
        figsize=(6, 6)
    )
    tabulate((fit_results_decay_const, fit_results_mass),
             observables,
             channels,
             'table7.tex')
