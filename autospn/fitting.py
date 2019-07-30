from numpy import (exp, outer, sum, asarray, swapaxes, mean, std, newaxis,
                   zeros_like, einsum)
from numpy.linalg import inv
from scipy.optimize import differential_evolution
from scipy.odr import ODR, Model, RealData
from scipy.stats import t


FITTING_INTENSITIES = {
    'default': {},
    'intense': {
        'popsize': 50,
        'tol': 0.001,
        'mutation': (0.5, 1.5),
        'recombination': 0.5
    }
}


def ps_fit_form(t, mass, decay_const, amplitude, NT):
    return amplitude ** 2 / mass * (
        exp(-mass * t) + exp(-mass * (NT - t))
    )


def ps_av_fit_form(t, mass, decay_const, amplitude, NT):
    return amplitude * decay_const * (
        exp(-mass * t) - exp(-mass * (NT - t))
    )


def v_fit_form(t, mass, decay_const, NT):
    return decay_const * mass * (
        exp(-mass * t) + exp(-mass * (NT - t))
    )


def odr_fit(f, x, y, xerr=None, yerr=None, p0=None, num_params=None):
    if not p0 and not num_params:
        raise ValueError("p0 or num_params must be specified")
    if p0 and (num_params is not None):
        assert len(p0) == num_params

    data_to_fit = RealData(x, y, xerr, yerr)
    model_to_fit_with = Model(f)
    if not p0:
        p0 = tuple(1 for _ in range(num_params))

    odr_analysis = ODR(data_to_fit, model_to_fit_with, p0)
    odr_analysis.set_job(fit_type=0)
    return odr_analysis.run()


def confpred_band(x, dfdp, fitobj, f,
                  prob=0.6321, abswei=False, err=None):
    '''From https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html
    Returns values for a confidence or a prediction band.
    '''
    # Given the confidence or prediction probability prob = 1-alpha
    # we derive alpha = 1 - prob
    alpha = 1 - prob
    prb = 1.0 - alpha / 2

    # Number of parameters from covariance matrix
    p = fitobj.beta
    n = len(p)
    if abswei:
        # Do not apply correction with red. chi^2
        covscale = 1.0
    else:
        covscale = fitobj.res_var

    dof = len(fitobj.xplus) - n
    tval = t.ppf(prb, dof)

    C = fitobj.cov_beta

    # df2_i = \sum_{jk} dfdp_ji dfdp_ki C_jk
    df2 = einsum('ji,ki,jk->i', dfdp, dfdp, C)

    if err is not None:
        df = (err * err + covscale * df2) ** 0.5
    else:
        df = (covscale * df2) ** 0.5
    y = f(p, x)
    delta = tval * df
    upperband = y + delta
    lowerband = y - delta
    return y, upperband, lowerband


def minimize_chisquare(correlator_sample_sets, mean_correlators, fit_functions,
                       parameter_ranges, plateau_start, plateau_end, NT,
                       fit_means=True, intensity='default'):
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
            **FITTING_INTENSITIES[intensity]
        )
        fit_results.append(fit_result.x)
        chisquare_values.append(fit_result.fun / degrees_of_freedom)

    return (tuple(zip(mean(fit_results, axis=0), std(fit_results, axis=0))),
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
