from numpy import exp, outer, sum, asarray, swapaxes, mean, std, newaxis
from numpy.linalg import inv
from scipy.optimize import differential_evolution


def ps_fit_form(t, mass, amplitude, decay_const, NT):
    return amplitude ** 2 / mass * (
        exp(-mass * t) + exp(-mass * (NT - t))
    )


def ps_av_fit_form(t, mass, amplitude, decay_const, NT):
    return amplitude * decay_const * (
        exp(-mass * t) - exp(-mass * (NT - t))
    )


def minimize_chisquare(correlator_sample_sets, mean_correlators, fit_functions,
                       plateau_start, plateau_end, NT, fit_means=True,
                       parameter_ranges=((0.01, 5), (0, 5), (0, 5))):
    assert (len(fit_functions) ==
            len(correlator_sample_sets) ==
            len(mean_correlators))
    degrees_of_freedom = (2 * (plateau_end - plateau_start) -
                          len(parameter_ranges))
    time_range = asarray(range(plateau_start, plateau_end))
    trimmed_mean_correlators = []
    inverse_covariances = []

    for sample_set, mean_correlator in zip(
            correlator_sample_sets, mean_correlators
    ):
        trimmed_mean_correlator = mean_correlator[plateau_start:plateau_end]
        trimmed_mean_correlators.append(trimmed_mean_correlator)
        covariance = (
            (sample_set[plateau_start:plateau_end] -
             trimmed_mean_correlator[:, newaxis]) @
            (sample_set[plateau_start:plateau_end] -
             trimmed_mean_correlator[:, newaxis]).T
        ) / (plateau_end - plateau_start) ** 2
        inverse_covariances.append(inv(covariance))
    if fit_means:
        sets_to_fit = (trimmed_mean_correlators,)
    else:
        sets_to_fit = swapaxes(
            asarray(tuple(
                swapaxes(correlator_sample_set[plateau_start:plateau_end],
                         0, 1)
                for correlator_sample_set in correlator_sample_sets)),
            0,
            1
        )

    fit_results = []
    chisquare_values = []

    for set_to_fit in sets_to_fit:
        fit_result = differential_evolution(
            chisquare,
            parameter_ranges,
            args=(
                set_to_fit,
                fit_functions,
                inverse_covariances,
                time_range,
                NT
            ),
            popsize=50,
            tol=0.001,
            mutation=(0.5, 1.5),
            recombination=0.5
        )
        fit_results.append(fit_result.x)
        chisquare_values.append(fit_result.fun / degrees_of_freedom)

    return (*zip(mean(fit_results, axis=0), std(fit_results, axis=0)),
            (mean(chisquare_values), std(chisquare_values)))


def chisquare(x, correlators_to_fit, fit_functions,
              inverse_covariances, time_range, NT):
    assert len(correlators_to_fit) == len(fit_functions)

    total_chisquare = 0

    for (correlator_to_fit, fit_function, inverse_covariance) in zip(
            correlators_to_fit, fit_functions, inverse_covariances
    ):
        difference_vector = (
            correlator_to_fit - fit_function(time_range, *x, NT)
        )
        total_chisquare += sum(
            outer(difference_vector, difference_vector) *
            inverse_covariance
        )
    return total_chisquare
