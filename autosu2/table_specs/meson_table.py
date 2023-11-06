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
    filename = "mesons.tex"

    generate_table_from_db(
        data=data,
        ensembles=ENSEMBLES,
        observables=observables,
        filename=filename,
        columns=columns,
        error_digits=ERROR_DIGITS,
        exponential=EXPONENTIAL,
        suppress_zeroes=True,
    )
