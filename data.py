from pandas import read_csv


def get_single_raw_correlator(all_correlators, channel, NT, NS, sign=+1,
                              ensemble_selection=0, initial_configuration=0,
                              configuration_separation=1):
    target_correlator = (
        all_correlators[all_correlators.channel == channel].iloc[
            (initial_configuration * configuration_separation +
             ensemble_selection)::
            configuration_separation
        ]
    )

    target_correlator.drop(['trajectory', 'channel'], axis=1, inplace=True)
    target_correlator[NT] = target_correlator[0]

    for column in range(NT // 2 + 1):
        target_correlator[column] += sign * target_correlator[NT - column]
        target_correlator[column] /= 2
    target_correlator.drop(range(NT // 2 + 1, NT + 1), axis=1, inplace=True)

    return NS ** 3 * target_correlator


def get_file_data(filename, NT):
    with open(filename) as f:
        number_of_columns = len(f.readline().split())

    assert NT == number_of_columns - 2

    column_names = ['trajectory', 'channel'] + list(range(NT))

    all_correlators = read_csv(filename, names=column_names,
                               delim_whitespace=True)

    return all_correlators


def get_target_correlator(filename, channels, NT, NS, signs,
                          ensemble_selection=0, initial_configuration=0,
                          configuration_separation=1):
    assert ensemble_selection < configuration_separation

    all_correlators = get_file_data(filename, NT)

    configuration_count = (len(all_correlators) // 17 //
                           configuration_separation - 1)

    used_configuration_count = configuration_count - initial_configuration + 1

    target_correlators = []

    for channel, sign in zip(channels, signs):
        target_correlators.append(get_single_raw_correlator(
            all_correlators, channel, NT, NS, sign,
            ensemble_selection, initial_configuration, configuration_separation
        ))

        assert (len(target_correlators[-1]) - 1 <=
                used_configuration_count <=
                len(target_correlators[-1]))

    return target_correlators
