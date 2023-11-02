#!/usr/bin/env python3

#!/usr/bin/env python

from functools import reduce
from operator import or_

import numpy as np

from ..tables import ObservableSpec
from ..db import get_measurement

observables = [
    ObservableSpec('w0c', free_parameter=0.2),
    ObservableSpec('w0p', free_parameter=0.2),
    ObservableSpec('s8t0c', free_parameter=0.2),
    ObservableSpec('s8t0p', free_parameter=0.2),
    ObservableSpec('Q0'),
    ObservableSpec('fitted_Q0'),
    ObservableSpec('Q_width'),
    ObservableSpec('Q_tau_exp'),
    ObservableSpec('mpcac_mass'),
    ObservableSpec('g5_mass'),
    ObservableSpec('g5_decay_const'),
    ObservableSpec('gk_mass'),
    ObservableSpec('id_mass'),
    ObservableSpec('g5gk_mass'),
    ObservableSpec('spin12_mass'),
]

def filter_observables(data, observables):
    masks = []
    for observable in observables:
        mask = (data.observable == observable.name)
        if observable.valence_mass:
            mask &= (data.valence_mass == observable.valence_mass)
        if observable.free_parameter:
            mask &= (data.free_parameter == observable.free_parameter)
        masks.append(mask)

    full_mask = reduce(or_, masks)
    return data[full_mask]


def filter_ensembles(data, ensembles):
    return data[data.label.isin(ensembles.keys())]


def serialise_key(multi_key):
    serial_key = multi_key[1]
    for key_component in multi_key[2:]:
        if not np.isnan(key_component):
            serial_key = f'{serial_key}_{key_component}'
    serial_key = f'{serial_key}_{multi_key[0]}'
    return serial_key


def generate(data, ensembles):
    filename = 'flow_data.csv'
    filtered_data = filter_ensembles(
        filter_observables(data, observables),
        ensembles,
    )
    tabular_data = filtered_data.pivot(
        index=['label', 'group_family', 'group_rank', 'representation', 'Nf',
               'L', 'T', 'beta', 'm', 'first_cfg', 'last_cfg', 'cfg_count'],
        columns=['observable', 'valence_mass', 'free_parameter']
    )
    tabular_data.columns = tabular_data.columns.map(serialise_key)
    tabular_data.sort_index(axis=1, inplace=True)
    tabular_data.to_csv(filename)
