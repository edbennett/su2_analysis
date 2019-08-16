from ..tables import generate_table_from_db

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


def table5(data):
    columns = ['', None, r'$am_{\mathrm{PS}}$', r'$af_{\mathrm{PS}}$',
               r'$am_{\mathrm{S}}$', None,
               r'$m_{\mathrm{PS}}L$', r'$f_{\mathrm{PS}}L$']
    observables = ('g5_mass', 'g5_renormalised_decay_const', 'id_mass',
                   'mPS_L', 'fPS_L')
    filename = 'table5.tex'

    for ensemble in ENSEMBLES:
        for source, dest in (('mass', 'mPS_L'),
                             ('renormalised_decay_const', 'fPS_L')):
            datum = data[(data.label == ensemble) &
                         (data.observable == f'g5_{source}')].copy()
            datum.observable = dest
            datum.value = datum.L * datum.value
            datum.uncertainty = datum.L * datum.uncertainty
            data = data.append(datum)

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )


def table6(data):
    columns = ['', None, r'$am_{\mathrm{V}}$', r'$af_{\mathrm{V}}$', None,
               r'$am_{\mathrm{AV}}$', r'$af_{\mathrm{AV}}$', None,
               r'$am_{\mathrm{T}}$', r'$am_{\mathrm{AT}}$']
    observables = ('gk_mass', 'gk_renormalised_decay_const', 'g5gk_mass',
                   'g5gk_renormalised_decay_const', 'g0gk_mass', 'g0g5gk_mass')
    filename = 'table6.tex'

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )


def generate(data, ensembles):
    table5(data)
    table6(data)
