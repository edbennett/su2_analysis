from ..tables import generate_table_from_db

from .ensembles import ENSEMBLES

ERROR_DIGITS = 2
EXPONENTIAL = False


def chunk_ensembles(ensembles, max_rows=45):
    current_chunk = []
    current_block = []

    for ensemble in ensembles:
        if ensemble is not None:
            current_block.append(ensemble)
            continue

        if len(current_chunk) + len(current_block) > max_rows:
            yield current_chunk
            current_chunk = current_block
        else:
            if current_chunk:
                current_chunk.append(None)
            current_chunk.extend(current_block)

        current_block = []
    else:
        if current_chunk:
            current_chunk.append(None)
        current_chunk.extend(current_block)
        yield current_chunk


def lattice_params(data, ensembles):
    columns = ["", None, r"$\beta$", "$am$", r"$N_t \times N_s^3$", r"$N_{\rm conf.}$"]
    constants = ("beta", "m", "V", "cfg_count")
    observables = []
    filename = "lattice_params_Nf{Nf}_part{chunk}.tex"

    for Nf, ensemble_set in ENSEMBLES.items():
        for chunk_index, ensemble_chunk in enumerate(chunk_ensembles(ensemble_set)):
            generate_table_from_db(
                data=data,
                ensemble_names=ensemble_chunk,
                observables=observables,
                filename=filename.format(Nf=Nf, chunk=chunk_index),
                columns=columns,
                constants=constants,
                error_digits=ERROR_DIGITS,
                exponential=EXPONENTIAL,
                ensembles_filename=ensembles["_filename"],
            )


def generate(data, ensembles):
    lattice_params(data, ensembles)
