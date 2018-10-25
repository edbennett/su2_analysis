from plots import do_eff_mass_plot, do_correlator_plot
from data import get_target_correlator
from bootstrap import bootstrap_correlators, bootstrap_eff_masses
from fitting import minimize_chisquare, ps_fit_form, ps_av_fit_form


def get_plot_filename(basename, type, channel='', tstart='', tend=''):
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

    return f'{basename}_{type}{channel}{tstart}{tend}.pdf'


def main():
    NT = 36
    NS = 28
    configuration_separation = 5
    initial_configuration = 10
    channel = 'g5'
    channel_latexes = (r'\gamma_5,\gamma_5', '\gamma_0\gamma_5,\gamma_5')
    bootstrap_sample_count = 200
    silent = False

    # ensemble_selection can range from 0  to configuration_separation
    ensemble_selection = 2

    filename = 'scratch/meson_corr_example/meson_corr_36x28x28x28b7.2m0.794'

    eff_mass_plot_ymin = 0.25
    eff_mass_plot_ymax = 0.5

    plateau_start = 13
    plateau_end = 18

    target_correlators = get_target_correlator(
        filename, (channel, 'g5_g0g5_re'), NT, NS, (+1, -1),
        ensemble_selection, initial_configuration, configuration_separation
    )
#    print(target_correlators[1].iloc[0:5])

    (bootstrap_mean_correlators, bootstrap_error_correlators,
     bootstrap_correlator_samples_set) = bootstrap_correlators(
         target_correlators, bootstrap_sample_count
     )
#    print(bootstrap_correlator_samples_set[1])

    bootstrap_mean_eff_masses, bootstrap_error_eff_masses = (
        bootstrap_eff_masses(bootstrap_correlator_samples_set)
    )

    do_eff_mass_plot(
        bootstrap_mean_eff_masses[0],
        bootstrap_error_eff_masses[0],
        get_plot_filename(filename, 'effmass', channel),
        ymin=eff_mass_plot_ymin,
        ymax=eff_mass_plot_ymax
    )

    for channel_to_plot, channel_latex, \
            bootstrap_mean_correlator, bootstrap_error_correlator in zip(
            (channel, 'g0g5_re'),
            channel_latexes,
            bootstrap_mean_correlators,
            bootstrap_error_correlators
            ):
        do_correlator_plot(
            bootstrap_mean_correlator,
            bootstrap_error_correlator,
            get_plot_filename(filename, 'correlator', channel),
            channel_latex
        )

    fit_results = minimize_chisquare(
        bootstrap_correlator_samples_set,
        bootstrap_mean_correlators,
        (ps_fit_form, ps_av_fit_form),
        plateau_start,
        plateau_end,
        NT,
        fit_means=True
    )
    ((mass, _), (amplitude, _),
     (decay_const, _), (chisquare_value, _)) = fit_results

    do_correlator_plot(
        bootstrap_mean_correlators[0],
        bootstrap_error_correlators[0],
        get_plot_filename(filename,
                          'centrally_fitted_correlator',
                          channel=channel,
                          tstart=plateau_start,
                          tend=plateau_end),
        channel_latexes[0],
        fit_function=ps_fit_form,
        fit_params=(mass, amplitude, decay_const, NT),
        fit_legend='Fit of central values',
        t_lowerbound=plateau_start - 3.5,
        t_upperbound=plateau_end - 0.5,
        corr_upperbound=0.002,
        corr_lowerbound=0.0
    )
    do_correlator_plot(
        bootstrap_mean_correlators[1],
        bootstrap_error_correlators[1],
        get_plot_filename(filename,
                          'centrally_fitted_correlator',
                          channel='g5_g0g5_re',
                          tstart=plateau_start,
                          tend=plateau_end),
        channel_latexes[1],
        fit_function=ps_av_fit_form,
        fit_params=(mass, amplitude, decay_const, NT),
        fit_legend='Fit of central values',
        t_lowerbound=plateau_start - 3.5,
        t_upperbound=plateau_end - 0.5,
        corr_upperbound=0.002,
        corr_lowerbound=0.0
    )

    fit_results = minimize_chisquare(
        bootstrap_correlator_samples_set,
        bootstrap_mean_correlators,
        (ps_fit_form, ps_av_fit_form),
        plateau_start,
        plateau_end,
        NT,
        fit_means=False
    )
    ((mass, mass_error), (amplitude, amplitude_error),
     (decay_const, decay_const_error),
     (chisquare_value, chisquare_error)) = fit_results

    do_eff_mass_plot(
        bootstrap_mean_eff_masses[0],
        bootstrap_error_eff_masses[0],
        get_plot_filename(filename,
                          'effmass_withfit',
                          channel=channel,
                          tstart=plateau_start,
                          tend=plateau_end),
        ymin=eff_mass_plot_ymin,
        ymax=eff_mass_plot_ymax,
        m=mass,
        m_error=mass_error,
        tmin=plateau_start - 0.5,
        tmax=plateau_end - 0.5
    )
    if not silent:
        print(f'{channel} mass: {mass} ± {mass_error}')
        print(f'{channel} decay constant: {decay_const} ± {decay_const_error}')
        print(f'{channel} amplitude: {amplitude} ± {amplitude_error}')
        print(f'{channel} chi-square: {chisquare_value} ± {chisquare_error}')


if __name__ == '__main__':
    main()
