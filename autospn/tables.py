from collections import namedtuple

from numpy import isnan
from uncertainties import ufloat

HLINE = r'    \hline'
ObservableSpec = namedtuple('ObservableSpec',
                            ('name', 'valence_mass', 'free_parameter'),
                            defaults=(None, None))


def table_row(row_content):
    return '    ' + ' & '.join(row_content)


def format_value_and_error(value, error, error_digits=2, exponential=False):
    if exponential:
        exp_flag = 'e'
    else:
        exp_flag = 'f'

    value = ufloat(value, error)
    format_string = f'${{:.{error_digits}u{exp_flag}SL}}$'
    return format_string.format(value)


def generate_table_from_content(columns, filename, table_content):
    table_spec = ''
    header = [column for column in columns if column is not None]
    table_spec = ''.join(['c' if column is not None else '|'
                          for column in columns])

    with open('final_tables/' + filename, 'w') as f:
        print(r'\begin{tabular}{' + table_spec + '}', file=f)
        print(table_row(header) + r' \\', file=f)
        print(HLINE, file=f)
        print(HLINE, file=f)
        print((r' \\' '\n').join(table_content), file=f)
        print(r'\end{tabular}', file=f)


def generate_table_from_db(
        data, ensembles, observables, filename, columns,
        constants=tuple(), error_digits=2, exponential=False
):
    table_content = []
    line_content = ''
    for ensemble in ensembles:
        if not ensemble:
            line_content += HLINE + '\n'
            continue

        ensemble_data = data[data.label == ensemble]
        if len(ensemble_data) == 0:
            print(f"WARNING: No data available for ensemble {ensemble}, "
                  "skipping")
            continue

        row_content = [ensemble]
        for constant in constants:
            value = set(ensemble_data[constant])
            assert len(value) == 1
            (value,) = value
            row_content.append(f'${str(value)}$')

        for observable in observables:
            if type(observable) == str:
                observable = ObservableSpec(observable)
            assert type(observable) == ObservableSpec
            measurement_mask = (
                (ensemble_data.observable == observable.name)
            )
            if (observable.free_parameter is None):
                measurement_mask = (
                    measurement_mask & ensemble_data.free_parameter.isnull()
                )
            else:
                measurement_mask = (
                    measurement_mask &
                    (ensemble_data.free_parameter == observable.free_parameter)
                )
            if (observable.valence_mass is None):
                measurement_mask = (
                    measurement_mask & ensemble_data.valence_mass.isnull()
                )
            else:
                measurement_mask = (
                    measurement_mask &
                    (ensemble_data.valence_mass == observable.valence_mass)
                )

            measurement = ensemble_data[measurement_mask]
            if len(measurement) > 1:
                import pdb; pdb.set_trace()
                raise ValueError(
                    "ensemble-observable combination is not unique"
                )

            if len(measurement) == 0:
                row_content.append('---')
                continue

            row_content.append(format_value_and_error(
                float(measurement.value),
                float(measurement.uncertainty),
                error_digits=error_digits,
                exponential=exponential
            ))
        line_content += table_row(row_content)
        table_content.append(line_content)
        line_content = ''

    generate_table_from_content(columns, filename, table_content)
