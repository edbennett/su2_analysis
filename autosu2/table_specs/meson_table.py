from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = [
        "",
        None,
        r"$am_{0^-_{\mathrm{v}}}$",
        r"$af_{0^-_{\mathrm{v}}}$",
        None,
        r"$am_{\breve{g}}$",
    ]
    observables = "gk_mass", "gk_decay_const", "spin12_mass"
    filename = "mesons_Nf{Nf}.tex"

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
