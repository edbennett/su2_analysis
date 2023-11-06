from ..tables import generate_table_from_db
from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def generate(data, **kwargs):
    columns = ["", None, r"$Q_0$", r"$\sigma$", None, r"$\tau_{\exp}$"]
    observables = ("fitted_Q0", "Q_width", "Q_tau_exp")
    filename = "topology_Nf{Nf}.tex"

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
