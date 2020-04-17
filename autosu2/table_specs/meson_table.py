from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES
ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ['', r'$m_{\mathrm{PCAC}}$', None, 
               r'$am_{\gamma_5}$', r'$af_{\gamma_5}$', None,
               r'$am_{\gamma_5\gamma_k}$', r'$af_{\gamma_5\gamma_k}$', None,
               r'$am_{1}$', r'$af_{1}$']
    observables = (
        'mpcac_mass', 'g5_mass', 'g5_decay_const', 'g5gk_mass',
        'g5gk_decay_const', 'id_mass', 'id_decay_const'
    )
    filename = 'mesons.tex'

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
