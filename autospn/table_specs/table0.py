from ..tables import generate_table_from_db, ObservableSpec

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


def generate(data, ensembles):
    columns = ['', None, r'$w_0^{\mathrm{p}}$', r'$w_0^{\mathrm{c}}$']
    observables = (ObservableSpec('w0p', free_parameter=0.35),
                   ObservableSpec('w0c', free_parameter=0.35))
    filename = 'table0.tex'

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )
