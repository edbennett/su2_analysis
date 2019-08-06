from ..tables import (
    generate_table_from_content, generate_table_from_db,
    ObservableSpec, HLINE, table_row
)
from ..db import get_measurement

ENSEMBLES = (
    'DB1M1', 'DB1M2', 'DB1M3', 'DB1M4', 'DB1M5', 'DB1M6',
    None,
    'DB2M1', 'DB2M2', 'DB2M3',
    None,
    'DB3M1', 'DB3M2', 'DB3M3', 'DB3M4', 'DB3M5', 'DB3M6', 'DB3M7', 'DB3M8',
    None,
    'DB4M1', 'DB4M2',
    None,
    'DB5M1'
)
ERROR_DIGITS = 2
EXPONENTIAL = False


def table1(data, **kwargs):
    columns = ['', None, r'$\beta$', '$am_0$', r'$N_t \times N_s^3$',
               r'$N_{\mathrm{configs}}$', r'$\delta_{\mathrm{traj}}$', None,
               r'$\langle P \rangle$', r'$w_0 / a$']
    constants = ('beta', 'm', 'V', 'Nconfigs', 'delta_traj')
    observables = ('avr_plaquette',
                   ObservableSpec('w0c', free_parameter=0.35),)
    filename = 'table1.tex'

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        constants=constants,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )


def table10(data, **kwargs):
    header = (r'\multirow{2}{c}{Ensemble} & '
              r'\multirow{2}{c}{$N_{\mathrm{configs}}$} & '
              r'\multirow{2}{c}{$\delta_{\mathrm{traj.}}$} & '
              r'\multicolumn{2}{c}{PS} & \multicolumn{2}{c}{V} & '
              r'\multicolumn{2}{c}{AV} & \multicolumn{2}{c}{S} \\ '
              r' & &'
              + r' & $I_{\mathrm{fit}}$ & $\chi^2$' * 4)
    table_spec = r'c|cc|cc|cc|cc|cc'
    ensembles = kwargs['ensembles']
    filename = 'table10.tex'

    table_content = []
    line_content = ''

    for ensemble_name in ENSEMBLES:
        if not ensemble_name:
            line_content += HLINE + '\n'
            continue

        ensemble = ensembles[ensemble_name]
        row_content = [ensemble_name,
                       str(ensemble['Nconfigs']), str(ensemble['delta_traj'])]
        meson_params = ensemble.get('measure_mesons', None)
        if not meson_params:
            row_content.extend(['---'] * 8)
            continue

        for channel_id, channel_name in (
                ('g5', 'PS'), ('gk', 'V'), ('g5gk', 'AV'), ('id', 'S')
        ):
            channel_params = meson_params.get(channel_id, None)
            if not channel_params:
                row_content.extend(['---', '---'])
                continue
            row_content.append(f'{channel_params["plateau_start"]}--'
                               f'{channel_params["plateau_end"]}')
            try:
                chisquare = get_measurement({'label': ensemble_name},
                                            f'{channel_id}_chisquare')
            except KeyError:
                row_content.append('---')
                continue
            else:
                row_content.append(f'{chisquare.value:.1f}')

        line_content += table_row(row_content)
        table_content.append(line_content)
        line_content = ''

    generate_table_from_content(filename, table_content,
                                header=header, table_spec=table_spec)


def generate(data, **kwargs):
    table1(data, **kwargs)
    table10(data, **kwargs)
