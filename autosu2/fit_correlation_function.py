from argparse import ArgumentParser
import logging

from meson_analysis.fits import (
    mean_multi_eff_mass,
    combine_multi_correlators,
    fit_multi_correlators,
)
from meson_analysis.fit_forms import get_fit_form
from meson_analysis.readers import read_correlators_hirep

from .plots import do_eff_mass_plot, do_correlator_plot, set_plot_defaults
from .data import get_output_filename
from .db import (
    measurement_is_up_to_date,
    add_measurement,
    purge_measurement,
)
from .bootstrap import (
    BOOTSTRAP_SAMPLE_COUNT,
)


channel_set_options = {
    "g5": (("g5",), ("g5_g0g5_re",)),
    "g5_mass": (("g5",),),
    "id": (("id",),),
    "gk": (("g1", "g2", "g3"),),
    "g5gk": (("g5g1", "g5g2", "g5g3"),),
    "g0gk": (("g0g1", "g0g2", "g0g3"),),
    "g0g5gk": (("g0g5g1", "g0g5g2", "g0g5g3"),),
}
correlator_names_options = {
    "g5": ("g5", "g5_g0g5_re"),
    "g5_mass": ("g5",),
    "id": ("id",),
    "gk": ("gk",),
    "g5gk": ("g5gk",),
    "g0gk": ("g0gk",),
    "g0g5gk": ("g0g5gk",),
}
channel_latexes_options = {
    "g5": (r"\gamma_5,\gamma_5", r"\gamma_0\gamma_5,\gamma_5"),
    "g5_mass": (r"\gamma_5,\gamma_5",),
    "id": (r"1,1",),
    "gk": (r"\gamma_k,\gamma_k",),
    "g5gk": (r"\gamma_5 \gamma_k,\gamma_5 \gamma_k",),
    "g0gk": (r"\gamma_0 \gamma_k,\gamma_0 \gamma_k",),
    "g0g5gk": (r"\gamma_0 \gamma_5 \gamma_k,\gamma_0 \gamma_5 \gamma_k",),
}
fit_forms_options = {
    "g5": ("ps", "ps_av"),
    "g5_mass": ("v",),
    "id": ("v",),
    "gk": ("v",),
    "g5gk": ("v",),
    "g0gk": ("v",),
    "g0g5gk": ("v",),
}
symmetries_options = {
    "g5": (+1, -1),
    "g5_mass": (+1,),
    "id": (+1,),
    "gk": (+1,),
    "g5gk": (+1,),
    "g0gk": (+1,),
    "g0g5gk": (+1,),
}
parameter_range_options = {
    "g5": ((0.01, 5), (0, 5), (0, 5)),
    "g5_mass": ((0.01, 5), (0, 5)),
    "id": ((0.01, 5), (0, 5)),
    "gk": ((0.01, 5), (0, 5)),
    "g5gk": ((0.01, 5), (0, 5)),
    "g0gk": ((0.01, 5), (0, 5)),
    "g0g5gk": ((0.01, 5), (0, 5)),
}
quantity_options = {
    "mpcac": ("mass", "chisquare"),
    "g5": ("mass", "decay_const", "amplitude", "chisquare"),
    "g5_mass": ("mass", "decay_const", "chisquare"),
    "id": ("mass", "decay_const", "chisquare"),
    "gk": ("mass", "decay_const", "chisquare"),
    "g5gk": ("mass", "decay_const", "chisquare"),
    "g0gk": ("mass", "decay_const", "chisquare"),
    "g0g5gk": ("mass", "decay_const", "chisquare"),
}


class Incomplete(Exception):
    pass


def process_correlator(
    correlator_filename,
    channel_name,
    channel_set,
    channel_latexes,
    symmetries,
    correlator_names,
    fit_forms,
    NT,
    NS,
    parameter_ranges,
    initial_configuration=0,
    bootstrap_sample_count=BOOTSTRAP_SAMPLE_COUNT,
    plateau_start=None,
    plateau_end=None,
    eff_mass_plot_ymin=None,
    eff_mass_plot_ymax=None,
    correlator_lowerbound=None,
    correlator_upperbound=None,
    optimizer_intensity="default",
    output_filename_prefix="",
    raw_correlators=True,
    _iter=0,
    maxiter=4,
):
    set_plot_defaults()

    correlators = read_correlators_hirep(correlator_filename)
    valence_masses = sorted(set(correlators.correlators.valence_mass))

    fit_results_set = []

    for valence_mass in valence_masses:
        try:
            eff_mass = mean_multi_eff_mass(
                correlators,
                channels=channel_set[0],
                parities={channel: symmetries[0] for channel in channel_set[0]},
                valence_mass=valence_mass,
            )
        except ValueError:
            logging.warn("pyerrors can't cope with this; skipping.")
            continue

        do_eff_mass_plot(
            eff_mass,
            get_output_filename(
                output_filename_prefix,
                "effmass",
                f"{valence_mass}_{correlator_names[0]}",
            ),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
        )

        for correlator_name, channel_latex, channel_names, symmetry in zip(
            correlator_names,
            channel_latexes,
            channel_set,
            symmetries,
        ):
            do_correlator_plot(
                combine_multi_correlators(
                    correlators,
                    channel_names,
                    valence_mass=valence_mass,
                    parity=symmetry,
                ),
                get_output_filename(
                    output_filename_prefix, "correlator", correlator_name
                ),
                channel_latex,
            )

        if not (plateau_start and plateau_end):
            continue

        result = fit_multi_correlators(
            correlators,
            [channel for channels in channel_set for channel in channels],
            [plateau_start, plateau_end],
            fit_forms={
                channel: get_fit_form(NT, fit_form)
                for channels, fit_form in zip(channel_set, fit_forms)
                for channel in channels
            },
            valence_mass=valence_mass,
            full=True,
        )
        fit_result_values = tuple(
            fit_result.value for fit_result in result.fit_parameters
        )

        for correlator_name, channel_latex, channel_name, fit_form, symmetry in zip(
            correlator_names,
            channel_latexes,
            channel_set,
            fit_forms,
            symmetries,
        ):
            do_correlator_plot(
                combine_multi_correlators(
                    correlators,
                    channel_names,
                    valence_mass=valence_mass,
                    parity=symmetry,
                ),
                get_output_filename(
                    output_filename_prefix,
                    "centrally_fitted_correlator",
                    channel=f"{valence_mass}_{correlator_name}",
                    tstart=plateau_start,
                    tend=plateau_end,
                ),
                channel_latex,
                fit_function=get_fit_form(NT, fit_form),
                fit_params={"params": fit_result_values, "NT": NT},
                fit_legend="Fit",
                t_lowerbound=plateau_start - 3.5,
                t_upperbound=plateau_end - 0.5,
                corr_upperbound=correlator_upperbound,
                corr_lowerbound=correlator_lowerbound,
            )

        fit_results_set.append((result.fit_parameters, result.chisquare / result.dof))

        do_eff_mass_plot(
            eff_mass,
            get_output_filename(
                output_filename_prefix,
                "effmass_withfit",
                channel=f"{valence_mass}_{correlator_names[0]}",
                tstart=plateau_start,
                tend=plateau_end,
            ),
            ymin=eff_mass_plot_ymin,
            ymax=eff_mass_plot_ymax,
            m=result.fit_parameters[0],
            tmin=plateau_start - 0.5,
            tmax=plateau_end + 0.5,
        )

    if not (plateau_start and plateau_end):
        raise Incomplete(
            "Effective mass plot has been generated. "
            "Now specify the start and end of the plateau to "
            "perform the fit."
        )

    return fit_results_set, valence_masses


def plot_measure_and_save_mesons(
    simulation_descriptor,
    correlator_filename,
    channel_name,
    output_filename_prefix=None,
    meson_parameters=None,
    parameter_date=None,
    force=False,
):
    # Distinguish between g5 with and without decay constant in analysis
    # But make them use same name in database to allow easy tabulation
    # and plotting
    db_channel_name = channel_name

    if not meson_parameters:
        meson_parameters = {}

    if not output_filename_prefix:
        output_filename_prefix = correlator_filename + "_"

    if meson_parameters.pop("no_decay_const", False) and channel_name == "g5":
        channel_name = "g5_mass"

    need_to_run = False
    if force:
        need_to_run = True
    else:
        for quantity_name in quantity_options[channel_name]:
            if not measurement_is_up_to_date(
                simulation_descriptor,
                f"{db_channel_name}_{quantity_name}",
                compare_file=correlator_filename,
            ):
                need_to_run = True
            if parameter_date and not measurement_is_up_to_date(
                simulation_descriptor,
                f"{db_channel_name}_{quantity_name}",
                compare_date=parameter_date,
            ):
                need_to_run = True
    if not need_to_run:
        return

    channel_set = channel_set_options[channel_name]
    correlator_names = correlator_names_options[channel_name]
    channel_latexes = channel_latexes_options[channel_name]
    fit_forms = fit_forms_options[channel_name]
    symmetries = symmetries_options[channel_name]
    parameter_ranges = parameter_range_options[channel_name]

    fit_results_set, valence_masses = process_correlator(
        correlator_filename,
        channel_name,
        channel_set,
        channel_latexes,
        symmetries,
        correlator_names,
        fit_forms,
        simulation_descriptor["T"],
        simulation_descriptor["L"],
        initial_configuration=simulation_descriptor.get("initial_configuration", 0),
        output_filename_prefix=output_filename_prefix,
        parameter_ranges=parameter_ranges,
        **meson_parameters,
    )

    if not force and len(valence_masses) > 0:
        if len(valence_masses) == 1:
            output_valence_masses = [None]
        else:
            output_valence_masses = valence_masses
        for valence_mass, values in zip(output_valence_masses, fit_results_set):
            values = values[0] + [
                values[1],
            ]
            for quantity_name, value in zip(quantity_options[channel_name], values):
                add_measurement(
                    simulation_descriptor,
                    f"{db_channel_name}_{quantity_name}",
                    value,
                    valence_mass=valence_mass,
                )
            if "decay_const" not in quantity_options[channel_name]:
                purge_measurement(
                    simulation_descriptor,
                    f"{db_channel_name}_decay_const",
                    valence_mass=valence_mass,
                )

    return valence_masses, fit_results_set


def main():
    parser = ArgumentParser()

    parser.add_argument("--correlator_filename", required=True)
    parser.add_argument("--channel", choices=("g5", "gk", "g5gk"), required=True)
    parser.add_argument("--NT", required=True, type=int)
    parser.add_argument("--NS", required=True, type=int)
    parser.add_argument("--initial_configuration", default=0, type=int)
    parser.add_argument("--bootstrap_sample_count", default=200, type=int)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument("--eff_mass_plot_ymin", default=None, type=float)
    parser.add_argument("--eff_mass_plot_ymax", default=None, type=float)
    parser.add_argument("--plateau_start", default=None, type=int)
    parser.add_argument("--plateau_end", default=None, type=int)
    parser.add_argument("--correlator_lowerbound", default=0.0, type=float)
    parser.add_argument("--correlator_upperbound", default=None, type=float)
    parser.add_argument(
        "--optimizer_intensity", default="default", choices=("default", "intense")
    )
    parser.add_argument("--output_filename_prefix", default=None)
    parser.add_argument("--ignore", action="store_true")
    parser.add_argument("--no_decay_const", action="store_true")
    parser.add_argument("--raw_correlators", action="store_true")
    args = parser.parse_args()

    meson_parameters = {
        key: args.__dict__[key]
        for key in [
            "eff_mass_plot_ymin",
            "eff_mass_plot_ymax",
            "plateau_start",
            "plateau_end",
            "correlator_lowerbound",
            "correlator_upperbound",
            "optimizer_intensity",
            "no_decay_const",
            "raw_correlators",
        ]
    }
    simulation_descriptor = {
        "L": args.NS,
        "T": args.NT,
    }
    if not args.ignore:
        try:
            valence_masses, fit_results_set = plot_measure_and_save_mesons(
                simulation_descriptor,
                args.correlator_filename,
                args.channel,
                output_filename_prefix=args.output_filename_prefix,
                meson_parameters=meson_parameters,
                force=True,
            )
        except Incomplete as ex:
            print("ANALYSIS NOT YET COMPLETE")
            print(ex)

        else:
            if not args.silent:
                for valence_mass, fit_results in zip(valence_masses, fit_results_set):
                    mass = fit_results[0][0]
                    decay_const = fit_results[0][1]
                    if len(fit_results[0]) > 2:
                        amplitude = fit_results[0][2]
                    chisquare_value = fit_results[1]

                    print(f"{args.channel} mass: {mass}")
                    print(f"{args.channel} decay constant: {decay_const}")
                    if len(fit_results[0]) > 2:
                        print(f"{args.channel} amplitude: {amplitude}")
                    print(f"{args.channel} chi-square: " f"{chisquare_value}")


if __name__ == "__main__":
    main()
