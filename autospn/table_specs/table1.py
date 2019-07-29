from ..tables import generate_table, ObservableSpec

ENSEMBLES = (
    'DB1M1', 'DB1M2', 'DB1M3', 'DB1M5', 'DB1M6', 'DB1M6',
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


def generate(data):
    header = ['', r'$\beta$', '$am_0$', r'$N_t \times N_s^3$',
              r'$N_{\mathrm{configs}}$', r'$\delta_{\mathrm{traj}}$',
              r'$\langle P \rangle$', r'$w_0 / a$']
    constants = ('beta', 'm', 'V', 'Nconfigs', 'delta_traj')
    observables = ('avr_plaquette',
                   ObservableSpec('w0c', free_parameter='0.35'),)
    filename = 'table1.tex'

    data['V'] = [
        f'{T} \\times {L}^3'
        for T, L in zip(data['T'], data['L'])
    ]

    generate_table(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        header=header,
        constants=constants,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )
