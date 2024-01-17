from contextlib import contextmanager
from datetime import datetime
from glob import glob
from os.path import getmtime

from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy import CheckConstraint, ForeignKey
from sqlalchemy import create_engine
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.hybrid import hybrid_property

from pyerrors import Obs
from uncertainties import UFloat, ufloat

from numpy import isnan
from pandas import read_sql

Base = declarative_base()

database_location = None
engine = None
Session = None


def init_engine(new_database_location="sqlite:///su2.sqlite"):
    # NB: Called automatically on module load, see below.

    global database_location, engine, Session

    database_location = new_database_location
    engine = create_engine(new_database_location)
    Session = sessionmaker(bind=engine)

    Base.metadata.create_all(engine)


class Simulation(Base):
    """A Simulation is a Monte Carlo ensemble generated at a particular set of
    parameters. Simulations should be unique."""

    __tablename__ = "simulation"
    id = Column(Integer, primary_key=True)
    label = Column(String)
    group_family = Column(String, nullable=False)
    group_rank = Column(Integer, nullable=False)
    representation = Column(String, nullable=True)
    Nf = Column(Integer, nullable=False)
    L = Column(Integer, nullable=False)
    T = Column(Integer, nullable=False)
    beta = Column(Float, nullable=False)
    m = Column(Float, nullable=True)
    first_cfg = Column(Integer, nullable=False)
    last_cfg = Column(Integer, nullable=False)
    cfg_count = Column(Integer, nullable=False)
    initial_configuration = Column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint("Nf >= 0", name="ck_nf_ge_0"),
        CheckConstraint(
            "((m != NULL) AND (representation != NULL) AND (Nf > 0)) OR "
            "((m == NULL) AND (representation == NULL) AND (Nf == 0))",
            name="ck_m_rep_match",
        ),
    )

    @hybrid_property
    def V(self):
        return self.L**3 * self.T

    @hybrid_property
    def is_quenched(self):
        return self.m is None


class Measurement(Base):
    """A Measurement is the value of a particular observable measured from a
    Monte Carlo ensemble (i.e. Simulation).
    Typically comes with an associated uncertainty.
    A Simulation typically has one measurement of each observable; however
    it may have multiple measurements at different `quenched_mass`es."""

    __tablename__ = "measurement"
    id = Column(Integer, primary_key=True)
    simulation_id = Column(Integer, ForeignKey("simulation.id"), nullable=False)
    simulation = relationship("Simulation")
    updated = Column(DateTime, nullable=False)
    observable = Column(String, nullable=False)
    valence_mass = Column(Float, nullable=True)
    # e.g. W0 for w0 measurement:
    free_parameter = Column(Float, nullable=True)
    value = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=True)


#    __table_args__ = (UniqueConstraint(
#        'simulation_id',
#        'observable',
#        'quenched_mass',
#        name='_each_observable_once'
#    ),)

init_engine()


@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations.
    Borrowed from sqlalchemy docs."""

    session = Session(expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_simulation(simulation_descriptor, session):
    """Query for a simulation matching the dictionary `simulation_descriptor`.
    If one is not found, then raise KeyError"""

    matching_simulation = (
        session.query(Simulation).filter_by(**simulation_descriptor).one_or_none()
    )
    if matching_simulation:
        return matching_simulation
    else:
        raise KeyError("Simulation not found")


def get_or_create_simulation(simulation_descriptor, session):
    """Query for a simulation matching the dictionary `simulation_descriptor`.
    If one is not found, then create one."""

    try:
        return get_simulation(simulation_descriptor, session)
    except KeyError:
        new_simulation = Simulation(**simulation_descriptor)
        session.add(new_simulation)
        return new_simulation


def get_measurement_inner(
    simulation, observable, valence_mass, free_parameter, session
):
    """Query for a measurement of `observable` (optionally at `quenched_mass`)
    for `simulation`.
    If one is found, return it.
    If none are found, raise KeyError.
    If more than one is found, raise sqlalchemy.orm.exc.MultipleResultsFound.
    """

    matching_measurement = (
        session.query(Measurement)
        .filter_by(
            simulation_id=simulation.id,
            observable=observable,
            valence_mass=valence_mass,
            free_parameter=free_parameter,
        )
        .one_or_none()
    )
    if matching_measurement:
        return matching_measurement
    else:
        raise KeyError("Measurement not found.")


def update_or_make_new_measurement(
    simulation, observable, valence_mass, free_parameter, value, uncertainty, session
):
    """Query for a measurement of `observable` (optionally at `valence_mass`)
    for `simulation`.
    If none is found, create one with `value` and `uncertainty`.
    If one is found, update it with `value` and `uncertainty`.
    If more than one is found, raise an error."""

    try:
        measurement = get_measurement_inner(
            simulation, observable, valence_mass, free_parameter, session
        )
    except KeyError:
        new_measurement = Measurement(
            simulation_id=simulation.id,
            updated=datetime.now(),
            observable=observable,
            valence_mass=valence_mass,
            free_parameter=free_parameter,
            value=value,
            uncertainty=uncertainty,
        )
        session.add(new_measurement)
        return new_measurement
    else:
        measurement.value = value
        measurement.uncertainty = uncertainty
        measurement.updated = datetime.now()
        return measurement


def add_measurement(
    simulation_descriptor,
    observable,
    value,
    uncertainty=None,
    valence_mass=None,
    free_parameter=None,
):
    """Add a new measurement to the database.
    Ensures that a simulation exists to associate the measurement with,
    then checks if an existing measurement exists, and if so updates it.
    Otherwise creates a new measurement and stores the result there."""

    if isinstance(value, UFloat):
        assert uncertainty is None
        uncertainty = value.std_dev
        value = value.nominal_value
    elif isinstance(value, Obs):
        assert uncertainty is None
        uncertainty = value.dvalue
        value = value.value

    if not isnan(value):
        with session_scope() as session:
            simulation = get_or_create_simulation(simulation_descriptor, session)
            update_or_make_new_measurement(
                simulation,
                observable,
                valence_mass,
                free_parameter,
                value,
                uncertainty,
                session,
            )


def get_measurement(
    simulation_descriptor, observable, valence_mass=None, free_parameter=None
):
    """Connect to the database and return the measurement of `observable` at
    `valence_mass` for a simulation described by `simulation_descriptor`."""

    with session_scope() as session:
        simulation = get_simulation(simulation_descriptor, session)
        return get_measurement_inner(
            simulation, observable, valence_mass, free_parameter, session
        )


def get_measurement_as_ufloat(
    simulation_descriptor, observable, valence_mass=None, free_parameter=None
):
    """Gets a measurement from the database, and turns it into a ufloat
    to allow it to be printed nicely."""

    measurement = get_measurement(
        simulation_descriptor, observable, valence_mass, free_parameter
    )
    return ufloat(measurement.value, measurement.uncertainty)


def measurement_exists(
    simulation_descriptor, observable, valence_mass=None, free_parameter=None
):
    """Check if a measurement of a particular `observable` and `valence mass`
    for a particular simulation described by `simulation_descriptor` exists.
    """
    try:
        if get_measurement(
            simulation_descriptor, observable, valence_mass, free_parameter
        ):
            return True
    except KeyError:
        return False


def single_simulation_exists(simulation_descriptor):
    """Check if a particular `simulation_descriptor` describes exactly one
    simulation in the database."""

    with session_scope() as session:
        try:
            get_simulation(simulation_descriptor, session)
        except Exception:
            return False
        else:
            return True


def measurement_is_up_to_date(
    simulation_descriptor,
    observable,
    valence_mass=None,
    free_parameter=None,
    compare_date=None,
    compare_dates=None,
    compare_glob=None,
    compare_file=None,
):
    """Check if a particular measurement is newer than `compare_date`,
    and the modification dates of the files matching `compare_glob`."""

    if not compare_dates:
        compare_dates = []
    if compare_date:
        compare_dates.append(compare_date)
    if compare_glob:
        compare_files = glob(compare_glob)
        if not compare_files:
            raise ValueError(f"Comparator files {compare_glob} missing.")
    else:
        compare_files = []

    if compare_file:
        compare_files.append(compare_file)

    if compare_files:
        compare_dates.extend([
            datetime.fromtimestamp(getmtime(compare_file))
            for compare_file in compare_files
        ])

    try:
        measurement = get_measurement(
            simulation_descriptor, observable, valence_mass, free_parameter
        )
    except KeyError:
        return False
    else:
        for single_date in compare_dates:
            if measurement.updated < single_date:
                return False
        else:
            return True


def purge_measurement(
    simulation_descriptor, observable, valence_mass=None, free_parameter=None
):
    """Connect to the database, and ensure that any measurement of a particular
    `observable` and `valence_mass` for a particular simulation described by
    `simulation_descrioptor` is removed from the database."""

    with session_scope() as session:
        try:
            simulation = get_simulation(simulation_descriptor, session)
            measurement = get_measurement_inner(
                simulation, observable, valence_mass, free_parameter, session
            )
        except KeyError:
            # Measurement didn't exist, so already purged
            return
        else:
            session.delete(measurement)


def is_complete_descriptor(simulation_descriptor):
    if (
        "group_family" in simulation_descriptor
        and "group_rank" in simulation_descriptor
        and "Nf" in simulation_descriptor
        and "L" in simulation_descriptor
        and "T" in simulation_descriptor
        and "beta" in simulation_descriptor
        and "first_cfg" in simulation_descriptor
        and "last_cfg" in simulation_descriptor
        and "cfg_count" in simulation_descriptor
    ):
        return True
    else:
        return False


def describe_ensemble(ensemble, label):
    """Returns a minimal dict describing an ensemble"""

    descriptor = {"label": label}
    for key in (
        "group_family",
        "group_rank",
        "Nf",
        "L",
        "T",
        "beta",
        "first_cfg",
        "last_cfg",
        "cfg_count",
    ):
        descriptor[key] = ensemble[key]
    for key in ("representation", "m", "initial_configuration"):
        if key in ensemble:
            descriptor[key] = ensemble[key]
    return descriptor


def get_dataframe():
    return read_sql(
        """SELECT measurement.observable,
               measurement.valence_mass,
               measurement.free_parameter,
               measurement.value,
               measurement.uncertainty,
               simulation.label,
               simulation.group_family,
               simulation.group_rank,
               simulation.representation,
               simulation.Nf,
               simulation.L,
               simulation.T,
               simulation.beta,
               simulation.m,
               simulation.first_cfg,
               simulation.last_cfg,
               simulation.cfg_count
        FROM measurement
        JOIN simulation
        ON measurement.simulation_id == simulation.id""",
        database_location,
    )
