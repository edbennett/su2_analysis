from argparse import ArgumentParser
import logging

from .plots import do_eff_mass_plot, do_correlator_plot, set_plot_defaults
from .data import get_target_correlator, get_output_filename
from .db import (
    measurement_is_up_to_date, add_measurement, measurement_exists,
    get_measurement, purge_measurement
)
from .fit_correlation_function import (
    channel_set_options, symmetries_options, quantity_options, Incomplete
)

from meson_analysis.fits import pcac_eff_mass, fit_pcac
from meson_analysis.readers import read_correlators_hirep


def process_mpcac(
        correlator_filename,
        plateau_start=None, plateau_end=None,
        eff_mass_plot_ymin=None, eff_mass_plot_ymax=None,
        correlator_lowerbound=None, correlator_upperbound=None,
        output_filename_prefix='',
):
    set_plot_defaults()

    correlators = read_correlators_hirep(correlator_filename)
    valence_masses = sorted(set(correlators.correlators.valence_mass))

    fit_results_set = []

    for valence_mass in valence_masses:
        try:
            eff_mass = pcac_eff_mass(correlators, valence_mass=valence_mass)
        except ValueError:
            logging.warn("pyerrors can't cope with this; skipping.")
            continue

        do_eff_mass_plot(
            eff_mass,
            get_output_filename(output_filename_prefix, 'effmass',
                                f'{valence_mass}_mpcac'),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax
        )

        if not (plateau_start and plateau_end):
            continue

        result = fit_pcac(correlators, [plateau_start, plateau_end], filters={"valence_mass": valence_mass}, full=True)

        fit_results_set.append((result[0], result.chisquare_by_dof))

        do_eff_mass_plot(
            eff_mass,
            get_output_filename(output_filename_prefix,
                                'effmass_withfit',
                                channel=f'{valence_mass}_mpcac',
                                tstart=plateau_start,
                                tend=plateau_end),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
            m=result[0],
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

    fit_results_set, valence_masses = process_mpcac(
        correlator_filename,
        output_filename_prefix=output_filename_prefix,
        **meson_parameters
    )

    if not force and len(valence_masses) > 0:
        if len(valence_masses) == 1:
            output_valence_masses = [None]
        else:
            output_valence_masses = valence_masses
        for valence_mass, values in zip(output_valence_masses,
                                        fit_results_set):
            add_measurement(simulation_descriptor, 'mpcac_mass', values[0],
                            valence_mass=valence_mass)
            add_measurement(simulation_descriptor, 'mpcac_chisquare', 
                            values[1], valence_mass=valence_mass)

    return valence_masses, fit_results_set


def main():
    parser = ArgumentParser()

    parser.add_argument('--correlator_filename', required=True)
    parser.add_argument('--NT', required=True, type=int)
    parser.add_argument('--NS', required=True, type=int)
    parser.add_argument('--initial_configuration', default=0, type=int)
    parser.add_argument('--bootstrap_sample_count', default=200, type=int)
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--eff_mass_plot_ymin', default=None, type=float)
    parser.add_argument('--eff_mass_plot_ymax', default=None, type=float)
    parser.add_argument('--plateau_start', default=None, type=int)
    parser.add_argument('--plateau_end', default=None, type=int)
    parser.add_argument('--correlator_lowerbound', default=0.0, type=float)
    parser.add_argument('--correlator_upperbound', default=None, type=float)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--ignore', action='store_true')
    args = parser.parse_args()

    meson_parameters = {
        key: args.__dict__[key] for key in [
            'eff_mass_plot_ymin', 'eff_mass_plot_ymax',
            'plateau_start', 'plateau_end',
            'correlator_lowerbound', 'correlator_upperbound',
        ]
    }
    simulation_descriptor = {
        'L': args.NS,
        'T': args.NT,
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
                    mass, chisquare_value = fit_results

                    print(f'PCAC mass: {mass}')
                    print(f'PCAC chi-square: {chisquare_value}')


if __name__ == '__main__':
    main()
