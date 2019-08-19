from ..tables import generate_table_from_db
from ..db import get_dataframe
from ..plot_specs.fig10_11 import EFT_ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def table8(data, **kwargs):
    columns = ['Ensemble', None, r'$\hat{a}$', None,
               r'$\hat{m}_\mathrm{PS}^2$', r'$\hat{f}_\mathrm{PS}^2$', None,
               r'$\hat{m}_\mathrm{V}^2$', r'$\hat{f}_\mathrm{V}^2$', None,
               r'$\hat{m}_\mathrm{AV}^2$', r'$\hat{f}_\mathrm{AV}^2$']
    observables = (
        'a_hat',
        'g5_mass_hat_squared',
        'g5_renormalised_decay_const_hat_squared_continuum_corrected',
        'gk_mass_hat_squared_continuum_corrected',
        'gk_renormalised_decay_const_hat_squared_continuum_corrected',
        'g5gk_mass_hat_squared_continuum_corrected',
        'g5gk_renormalised_decay_const_hat_squared_continuum_corrected',
    )
    filename = 'table8.tex'

    generate_table_from_db(
        data=data,
        ensembles=EFT_ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL
    )


def generate(data, **kwargs):
    data = get_dataframe()
    table8(data, **kwargs)
