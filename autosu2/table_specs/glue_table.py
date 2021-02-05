from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES
ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ['', r'$a \sqrt{\sigma}$', None,
               r'$am_{0^{++}}$', r'$am_{2^{++}}$']
    observables = 'sqrtsigma', 'App_mass', 'Epp_mass'
    filename = 'glue.tex'

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL,
        suppress_zeroes=True,
        skip_empty_rows=True
    )
