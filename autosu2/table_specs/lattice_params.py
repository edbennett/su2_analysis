from ..tables import generate_table_from_db

from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def lattice_params(data, **kwargs):
    columns = ["", None, r"$\beta$", "$am$", r"$N_t \times N_s^3$", r"$N_{\rm conf.}$"]
    constants = ("beta", "m", "V", "cfg_count")
    observables = []
    filename = "lattice_params_Nf{Nf}.tex"

    for Nf, ensemble_set in ENSEMBLES.items():
        generate_table_from_db(
            data=data,
            ensembles=ensemble_set,
            observables=observables,
            filename=filename.format(Nf=Nf),
            columns=columns,
            constants=constants,
            error_digits=ERROR_DIGITS,
            exponential=EXPONENTIAL,
        )


def generate(data, **kwargs):
    lattice_params(data, **kwargs)
