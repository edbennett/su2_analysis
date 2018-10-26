from numpy import ndarray, mean, std, arccosh
from numpy.random import randint


def bootstrap_correlators(target_correlators, bootstrap_sample_count=200):
    assert len(target_correlators) > 0
    assert len(set(map(len, target_correlators))) == 1

    number_of_configurations = target_correlators[0].shape[0]
    NT = target_correlators[0].shape[1] * 2

    bootstrap_sample_configurations = randint(
        number_of_configurations,
        size=(number_of_configurations, bootstrap_sample_count)
    )

#    print(bootstrap_sample_configurations)

    bootstrap_correlator_samples_set = []
    bootstrap_mean_correlators = []
    bootstrap_error_correlators = []

    for target_correlator in target_correlators:
        bootstrap_correlator_samples_set.append(
            ndarray((NT // 2, bootstrap_sample_count))
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
