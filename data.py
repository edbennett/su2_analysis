from pandas import read_csv, concat
from numpy import asarray
from csv import writer, QUOTE_MINIMAL


def write_results(filename, headers, values_set, channel_name='',
                  many=False, extras_set=None):
    if channel_name:
        channel_name = f'{channel_name}_'
    if not many:
        values_set = (values_set,)
    if not extras_set:
        extras_set = [[]] * len(values_set)

    with open(filename, 'w', newline='') as csvfile:
        csv_writer = writer(csvfile,
                            delimiter='\t',
                            quoting=QUOTE_MINIMAL,
                            lineterminator='\n')
        csv_writer.writerow((f'{channel_name}{header}'
                             for header in headers))
#        import pdb; pdb.set_trace()
        for values, extras in zip(values_set, extras_set):
            csv_writer.writerow([value
                                 for value_pair in values
                                 for value in value_pair] + extras)


def read_db(filename):
    return read_csv(filename, delim_whitespace=True, na_values='?')


def get_output_filename(basename, type, channel='', tstart='', tend='',
                        filetype='pdf'):
    if channel:
        channel = f'_{channel}'
    if tstart:
        tstart = f'_{tstart}'
    if tend:
        tend = f'_{tend}'
    if tstart and not tend:
        tend = '_XX'
    if tend and not tstart:
        tstart = '_XX'

    return f'{basename}{type}{channel}{tstart}{tend}.{filetype}'


def get_single_raw_correlator_set(all_correlators, channels, NT, NS,
                                  valence_masses,
                                  sign=+1,
                                  ensemble_selection=0,
                                  initial_configuration=0,
                                  configuration_separation=1):
    target_correlator = concat(
        (
            all_correlators[all_correlators.channel == channel].iloc[
                (initial_configuration * configuration_separation +
                 ensemble_selection)::
                configuration_separation
            ]
            for channel in channels
        )
    )

    target_correlator.drop(['trajectory', 'channel'], axis=1, inplace=True)
    target_correlator[NT] = target_correlator[0]

    for column in range(NT // 2 + 1):
        target_correlator[column] += sign * target_correlator[NT - column]
        target_correlator[column] /= 2
    target_correlator.drop(range(NT // 2 + 1, NT + 1), axis=1, inplace=True)

    correlator_set = []

    for valence_mass in valence_masses:
        correlator_set.append(
            target_correlator[
                target_correlator.valence_mass == valence_mass
            ].drop(['valence_mass'], axis=1) * NS ** 3
        )

    return correlator_set


def get_file_data(filename, NT):
    with open(filename) as f:
        number_of_columns = len(f.readline().split())

    assert NT == number_of_columns - 3

    column_names = ['trajectory', 'valence_mass', 'channel'] + list(range(NT))

    all_correlators = read_csv(filename, names=column_names,
                               delim_whitespace=True)

    return all_correlators


def get_target_correlator(filename, channel_sets, NT, NS, signs,
                          ensemble_selection=0, initial_configuration=0,
                          configuration_separation=1):
    assert ensemble_selection < configuration_separation

    all_correlators = get_file_data(filename, NT)
    valence_masses = sorted(set(all_correlators.valence_mass))

    configuration_count = (len(all_correlators) // 17 //
                           configuration_separation - 1)

    used_configuration_count = configuration_count - initial_configuration + 1

    target_correlator_sets = []

    for channels, sign in zip(channel_sets, signs):
        target_correlator_sets.append(get_single_raw_correlator_set(
            all_correlators, channels, NT, NS, valence_masses, sign,
            ensemble_selection, initial_configuration, configuration_separation
        ))
        for raw_correlator in target_correlator_sets[-1]:
            assert (len(raw_correlator) - len(channels) <=
                    used_configuration_count * len(channels) <=
                    len(raw_correlator))
        assert len(target_correlator_sets[-1]) == len(valence_masses)

    return map(list, zip(*target_correlator_sets)), valence_masses


def get_flows(filename):
    data = read_csv(filename, delim_whitespace=True,
                    names=['n', 't', 'Ep', 'Ec', 'Q'])
    times = asarray(sorted(set(data.t)))
    Eps = asarray(
        [
            data[data.n == n].Ep.values
            for n in set(data.n)
        ]
    )
    Ecs = asarray(
        [
            data[data.n == n].Ec.values
            for n in set(data.n)
        ]
    )

    return times, Eps, Ecs
