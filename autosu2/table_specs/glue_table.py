from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ["", r"$a \sqrt{\sigma}$", None, r"$am_{0^{++}}$", r"$am_{2^{++}}$"]
    observables = "sqrtsigma", "App_mass", "Epp_mass"
    filename = "glue_Nf{Nf}.tex"

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
            skip_empty_rows=True,
        )
