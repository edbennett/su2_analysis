from numpy import ndarray, mean, std, arccosh, asarray
from numpy.random import randint, choice

BOOTSTRAP_SAMPLE_COUNT = 200


def basic_bootstrap(values):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        samples.append(mean(choice(values, len(values))))
    return mean(samples, axis=0), std(samples)


def bootstrap_susceptibility(values):
    values = asarray(values)
    samples = []
    for _ in range(BOOTSTRAP_SAMPLE_COUNT):
        current_sample = choice(values, len(values))
        samples.append(mean(current_sample ** 2) - mean(current_sample) ** 2)
    return mean(samples, axis=0), std(samples)


def bootstrap_1d(values):
    values = asarray(values)
    bootstrap_sample_configurations = randint(
        values.shape[0],
        size=(values.shape[0], BOOTSTRAP_SAMPLE_COUNT)
    )
    bootstrap_samples = []
    for t_index in range(values.shape[1]):
        bootstrap_samples.append(
            values[bootstrap_sample_configurations, t_index].mean(axis=1)
        )
    bootstrap_samples = asarray(bootstrap_samples)
    return mean(bootstrap_samples, axis=1), std(bootstrap_samples, axis=1)


def bootstrap_correlators(target_correlators):
    assert len(target_correlators) > 0
    assert len(set(map(len, target_correlators))) == 1

    number_of_configurations = target_correlators[0].shape[0]
    NT = target_correlators[0].shape[1] * 2

    bootstrap_sample_configurations = randint(
        number_of_configurations,
        size=(number_of_configurations, BOOTSTRAP_SAMPLE_COUNT)
    )

#    print(bootstrap_sample_configurations)

    bootstrap_correlator_samples_set = []
    bootstrap_mean_correlators = []
    bootstrap_error_correlators = []

    for target_correlator in target_correlators:
        bootstrap_correlator_samples_set.append(
            ndarray((NT // 2, BOOTSTRAP_SAMPLE_COUNT))
        )
        for timeslice in range(NT // 2):
            bootstrap_correlator_samples_set[-1][timeslice] = (
                target_correlator[timeslice].values[
                    bootstrap_sample_configurations
                ].mean(axis=0)
            )
#        print(bootstrap_correlator_samples_set[-1])

        bootstrap_mean_correlators.append(
            bootstrap_correlator_samples_set[-1].mean(axis=1)
        )
        bootstrap_error_correlators.append(
            bootstrap_correlator_samples_set[-1].std(axis=1)
        )

    return (bootstrap_mean_correlators, bootstrap_error_correlators,
            bootstrap_correlator_samples_set)


def bootstrap_eff_masses(bootstrap_correlator_samples_set):
    if len(bootstrap_correlator_samples_set) > 1:
        assert len(set(samples.shape
                       for samples in bootstrap_correlator_samples_set)) == 1
    eff_mass_samples_shape = list(bootstrap_correlator_samples_set[0].shape)
    eff_mass_samples_shape[0] -= 2

    bootstrap_mean_eff_masses = []
    bootstrap_error_eff_masses = []
    for bootstrap_correlator_samples in bootstrap_correlator_samples_set:
        eff_mass_samples = ndarray(eff_mass_samples_shape)

        for timeslice in range(eff_mass_samples_shape[0]):
            eff_mass_samples[timeslice] = arccosh(
                (bootstrap_correlator_samples[timeslice] +
                 bootstrap_correlator_samples[timeslice + 2]) /
                (2 * bootstrap_correlator_samples[timeslice + 1])
            )

        bootstrap_mean_eff_masses.append(
            mean(eff_mass_samples, axis=1)
        )
        bootstrap_error_eff_masses.append(
            std(eff_mass_samples, axis=1)
        )

    return bootstrap_mean_eff_masses, bootstrap_error_eff_masses
