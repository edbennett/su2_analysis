from argparse import ArgumentParser

from .plots import do_eff_mass_plot, do_correlator_plot, set_plot_defaults
from .data import get_target_correlator, get_output_filename
from .db import (
    measurement_is_up_to_date, add_measurement, measurement_exists,
    get_measurement, purge_measurement
)
from .bootstrap import (bootstrap_correlators, bootstrap_pcac_eff_mass,
                        BOOTSTRAP_SAMPLE_COUNT)
from .fitting import minimize_chisquare
from .fit_correlation_function import (
    channel_set_options, symmetries_options, quantity_options, Incomplete
)


def process_mpcac(
        correlator_filename,
        NT, NS,
        initial_configuration=0, bin_size=1,
        bootstrap_sample_count=BOOTSTRAP_SAMPLE_COUNT,
        plateau_start=None, plateau_end=None,
        eff_mass_plot_ymin=None, eff_mass_plot_ymax=None,
        correlator_lowerbound=None, correlator_upperbound=None,
        output_filename_prefix='',
        raw_correlators=True
):
    set_plot_defaults()

    target_correlator_sets, valence_masses = get_target_correlator(
        correlator_filename, channel_set_options['g5'], NT, NS,
        symmetries_options['g5'], initial_configuration, bin_size=bin_size,
        from_raw=raw_correlators
    )

    fit_results_set = []

    for target_correlators, valence_mass in zip(
            target_correlator_sets, valence_masses
    ):
        (bootstrap_mean_correlators, bootstrap_error_correlators,
         bootstrap_correlator_samples_set) = bootstrap_correlators(
             target_correlators
         )

        (
            bootstrap_mean_eff_mass, bootstrap_error_eff_mass,
            mass, mass_error, chisquare
        ) = bootstrap_pcac_eff_mass(
            bootstrap_correlator_samples_set,
            plateau_start,
            plateau_end
        )

        do_eff_mass_plot(
            bootstrap_mean_eff_mass,
            bootstrap_error_eff_mass,
            get_output_filename(output_filename_prefix, 'effmass',
                                f'{valence_mass}_mpcac'),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax
        )

        if not (plateau_start and plateau_end):
            continue

        fit_results_set.append(((mass, mass_error), chisquare))

        do_eff_mass_plot(
            bootstrap_mean_eff_mass,
            bootstrap_error_eff_mass,
            get_output_filename(output_filename_prefix,
                                'effmass_withfit',
                                channel=f'{valence_mass}_mpcac',
                                tstart=plateau_start,
                                tend=plateau_end),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
            m=mass,
            m_error=mass_error,
            tmin=plateau_start - 0.5,
            tmax=plateau_end - 0.5
        )

    if not (plateau_start and plateau_end):
        raise Incomplete(
            "Effective mass plot has been generated. "
            "Now specify the start and end of the plateau to "
            "perform the fit."
        )
    return fit_results_set, valence_masses


def plot_measure_and_save_mpcac(simulation_descriptor, correlator_filename,
                                output_filename_prefix=None,
                                meson_parameters=None,
                                parameter_date=None,
                                force=False):
    if not meson_parameters:
        meson_parameters = {}

    if not output_filename_prefix:
        output_filename_prefix = correlator_filename + '_'

    need_to_run = False
    if force:
        need_to_run = True
    else:
        for quantity_name in quantity_options['mpcac']:
            if not measurement_is_up_to_date(
                    simulation_descriptor,
                    f'mpcac_{quantity_name}',
                    compare_file=correlator_filename
            ):
                need_to_run = True
            if parameter_date and not measurement_is_up_to_date(
                    simulation_descriptor,
                    f'mpcac_{quantity_name}',
                    compare_date=parameter_date
            ):
                need_to_run = True
    if not need_to_run:
        return

    if simulation_descriptor and measurement_exists(
            simulation_descriptor, 'Q_tau_exp'
    ):
        bin_size = int(
            get_measurement(simulation_descriptor, 'Q_tau_exp').value
        ) + 1

    fit_results_set, valence_masses = process_mpcac(
        correlator_filename,
        simulation_descriptor['T'],
        simulation_descriptor['L'],
        initial_configuration=simulation_descriptor.get(
            'initial_configuration', 0
        ),
        output_filename_prefix=output_filename_prefix,
        bin_size=bin_size,
        **meson_parameters
    )

    if not force and len(valence_masses) > 0:
        if len(valence_masses) == 1:
            output_valence_masses = [None]
        else:
            output_valence_masses = valence_masses
        for valence_mass, values in zip(output_valence_masses,
                                        fit_results_set):
            add_measurement(simulation_descriptor, 'mpcac_mass', *values[0],
                            valence_mass=valence_mass)
            add_measurement(simulation_descriptor, 'mpcac_chisquare', 
                            values[1], valence_mass=valence_mass)

    return valence_masses, fit_results_set


def main():
    parser = ArgumentParser()

    parser.add_argument('--correlator_filename', required=True)
    parser.add_argument('--channel', choices=('g5', 'gk', 'g5gk'),
                        required=True)
    parser.add_argument('--NT', required=True, type=int)
    parser.add_argument('--NS', required=True, type=int)
    parser.add_argument('--bin_size', default=1, type=int)
    parser.add_argument('--initial_configuration', default=0, type=int)
    parser.add_argument('--bootstrap_sample_count', default=200, type=int)
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--eff_mass_plot_ymin', default=None, type=float)
    parser.add_argument('--eff_mass_plot_ymax', default=None, type=float)
    parser.add_argument('--plateau_start', default=None, type=int)
    parser.add_argument('--plateau_end', default=None, type=int)
    parser.add_argument('--correlator_lowerbound', default=0.0, type=float)
    parser.add_argument('--correlator_upperbound', default=None, type=float)
    parser.add_argument('--optimizer_intensity', default='default',
                        choices=('default', 'intense'))
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--ignore', action='store_true')
    parser.add_argument('--no_decay_const', action='store_true')
    parser.add_argument('--raw_correlators', action='store_true')
    args = parser.parse_args()

    meson_parameters = {
        key: args.__dict__[key] for key in [
            'eff_mass_plot_ymin', 'eff_mass_plot_ymax',
            'plateau_start', 'plateau_end',
            'correlator_lowerbound', 'correlator_upperbound',
            'optimizer_intensity',
            'no_decay_const',
            'raw_correlators'
        ]
    }
    simulation_descriptor = {
        'L': args.NS,
        'T': args.NT,
        'delta_traj': args.bin_size,
        'initial_configuration': args.initial_configuration
    }
    if not args.ignore:
        try:
            valence_masses, fit_results_set = plot_measure_and_save_mpcac(
                simulation_descriptor,
                args.correlator_filename,
                output_filename_prefix=args.output_filename_prefix,
                meson_parameters=meson_parameters,
                force=True
            )
        except Incomplete as ex:
            print("ANALYSIS NOT YET COMPLETE")
            print(ex)

        else:
            if not args.silent:
                for valence_mass, fit_results in zip(
                        valence_masses, fit_results_set
                ):
                    mass, mass_error = fit_results[0][0]
                    chisquare_value = fit_results[1]

                    print(f'{args.channel} mass: {mass} ± {mass_error}')
                    print(f'{args.channel} chi-square: '
                          f'{chisquare_value}')


if __name__ == '__main__':
    main()
