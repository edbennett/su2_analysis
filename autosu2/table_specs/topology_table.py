from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES
ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ['', None, r'$Q_0$', r'$\sigma$', None,
               r'$\tau_{\exp}$']
    observables = ('fitted_Q0', 'Q_width', 'Q_tau_exp')
    filename = 'topology.tex'

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL,
        suppress_zeroes=True
    )
