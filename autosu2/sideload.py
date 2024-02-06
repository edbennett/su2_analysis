import logging
from pathlib import Path

from numpy import isnan
from sqlalchemy import bindparam, create_engine, text

from .db import add_measurement, measurement_exists, single_simulation_exists


def import_data(data, observables):
    """
    Import specified observables from a Pandas DataFrame into the database.
    """

    for _, datum in data.iterrows():
        ensemble_label = datum.Ensemble
        if not single_simulation_exists({"label": ensemble_label}):
            print(
                f"WARNING: simulation {ensemble_label} is not in the "
                "database. The database should have all simulations in "
                "prior to importing external data."
            )
            continue

        for observable in observables:
            q_value = datum[f"{observable}_value"]
            q_uncertainty = datum[f"{observable}_uncertainty"]
            if not (isnan(q_value) or isnan(q_uncertainty)):
                add_measurement(
                    simulation_descriptor={"label": ensemble_label},
                    observable=observable,
                    value=q_value,
                    uncertainty=q_uncertainty,
                )


def describe_ensemble(measurement):
    keys = (
        "label",
        "group_family",
        "group_rank",
        "representation",
        "Nf",
        "L",
        "T",
        "beta",
        "m",
        "first_cfg",
        "last_cfg",
        "cfg_count",
        "initial_configuration",
    )
    return {key: getattr(measurement, key) for key in keys}


def check_ensemble_matches(measurement):
    descriptor = describe_ensemble(measurement)
    if not single_simulation_exists(descriptor):
        raise ValueError(
            f"No existing ensembles match for ensemble {measurement.label}"
        )


class SelfMap:
    def __getitem__(self, key):
        return key


def callback_string_tension(measurement):
    if measurement.observable == "sqrtsigma":
        add_measurement(
            describe_ensemble(measurement),
            "string_tension",
            measurement.value**2,
            uncertainty=2 * measurement.value * measurement.uncertainty,
            valence_mass=measurement.valence_mass,
            free_parameter=measurement.free_parameter,
        )


def import_data_sql(
    filename, ensembles, observables, observable_names={}, callback=None
):
    if not Path(filename).exists():
        raise ValueError(f"No database at {filename} to import from.")

    engine = create_engine("sqlite:///" + str(Path(filename).absolute()))
    with engine.connect() as conn:
        for measurement in conn.execute(
            text(
                "SELECT * from measurement "
                "JOIN simulation ON simulation.id = measurement.simulation_id "
                "WHERE label IN :ensembles AND observable IN :observables"
            ).bindparams(
                bindparam(key="ensembles", value=ensembles, expanding=True),
                bindparam(key="observables", value=observables, expanding=True),
            )
        ):
            check_ensemble_matches(measurement)
            descriptor = describe_ensemble(measurement)
            if measurement_exists(
                descriptor,
                measurement.observable,
                valence_mass=measurement.valence_mass,
                free_parameter=measurement.free_parameter,
            ):
                logging.warn(
                    f"Measurement of {measurement.observable} already present "
                    f"for {measurement.label}; not overwriting"
                )
                continue

            add_measurement(
                descriptor,
                observable_names.get(
                    measurement.observable, SelfMap()[measurement.observable]
                ),
                measurement.value,
                uncertainty=measurement.uncertainty,
                valence_mass=measurement.valence_mass,
                free_parameter=measurement.free_parameter,
            )
            if callback:
                callback(measurement)
