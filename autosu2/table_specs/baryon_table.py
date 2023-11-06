from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = [
        "",
        r"$m_{\mathrm{PCAC}}$",
        None,
        r"$am_{2^+_{\mathrm{ps}}}$",
        r"$af_{2^+_{\mathrm{ps}}}$",
        None,
        r"$am_{2^-_{\mathrm{v}}}$",
        r"$af_{2^-_{\mathrm{v}}}$",
        None,
        r"$am_{2^-_{\mathrm{ps}}}$",
        r"$af_{2^-_{\mathrm{ps}}}$",
    ]
    observables = (
        "mpcac_mass",
        "g5_mass",
        "g5_decay_const",
        "g5gk_mass",
        "g5gk_decay_const",
        "id_mass",
        "id_decay_const",
    )
    filename = "baryons_Nf{Nf}.tex"

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
