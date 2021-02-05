from numpy import isnan
from .db import add_measurement, single_simulation_exists

def import_data(data, observables):
    '''
    Import specified observables from a Pandas DataFrame into the database.
    '''

    for _, datum in data.iterrows():
        ensemble_label = datum.Ensemble
        if not single_simulation_exists({'label': ensemble_label}):
            print(f"WARNING: simulation {ensemble_label} is not in the "
                  "database. The database should have all simulations in "
                  "prior to importing external data.")
            continue

        for observable in observables:
            q_value = datum[f'{observable}_value']
            q_uncertainty = datum[f'{observable}_uncertainty']
            if not (isnan(q_value) or isnan(q_uncertainty)):
                add_measurement(
                    simulation_descriptor={'label': ensemble_label},
                    observable=observable,
                    value=q_value,
                    uncertainty=q_uncertainty
                )
