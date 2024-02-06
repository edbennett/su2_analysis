from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ["ensemble", r"$\beta$", r"$\gamma^*$ (from $\nu$)"]
    observables = ("gamma_aic", "gamma_aic_syst")
    filename = "gamma_modenumber.tex"

    for Nf, ensemble_set in ENSEMBLES.items():
        generate_table_from_db(
            data=data,
            ensembles=ensemble_set,
            observables=observables,
            filename=filename.format(Nf=Nf),
            columns=columns,
            error_digits=ERROR_DIGITS,
            exponential=EXPONENTIAL,
            suppress_zeroes=True,
        )
