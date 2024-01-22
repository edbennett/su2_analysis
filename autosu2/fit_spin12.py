#!/usr/bin/env python3

from pathlib import Path

from meson_analysis.correlator import CorrelatorEnsemble
from meson_analysis.fits import fit_single_correlator
from meson_analysis.fit_forms import get_fit_form
from meson_analysis.readers import read_correlators_flexlatsim

from .plots import do_eff_mass_plot, do_correlator_plot, set_plot_defaults
from .data import get_output_filename
from .db import measurement_is_up_to_date, add_measurement
from .fit_correlation_function import Incomplete


def get_correlators(directory_name, valence_mass, NT):
    correlators = CorrelatorEnsemble(directory_name)
    directory = Path(directory_name)
    max_num_streams = 3

    if (filename := directory / "out_corr_spin12").exists():
        to_process = [("", filename)]
    else:
        to_process = []
        for stream_idx in range(1, max_num_streams + 1):
            if not (filename := directory / f"out_corr_spin12_{stream_idx}").exists():
                break
            to_process.append((stream_idx, filename))

    for stream_idx, filename in to_process:
        read_correlators_flexlatsim(
            filename,
            valence_mass,
            correlators=correlators,
            freeze=False,
            stream_name=f"run{stream_idx}",
            NT=NT,
        )

    correlators.freeze()
    return correlators


def process_correlator(
    directory_name,
    NT,
    valence_mass,
    plateau_start=None,
    plateau_end=None,
    eff_mass_plot_ymin=None,
    eff_mass_plot_ymax=None,
    correlator_lowerbound=None,
    correlator_upperbound=None,
    output_filename_prefix="",
):
    set_plot_defaults()
    correlators = get_correlators(directory_name, NT, valence_mass)

    correlator = correlators.get_pyerrors(channel="GGCorrTr1", symmetry="antisymmetric")
    correlator.gamma_method()

    try:
        eff_mass = correlator.m_eff(variant="cosh")
        eff_mass.gamma_method()
    except ValueError:
        logging.warn("pyerrors can't cope with this; skipping.")
        return

    do_eff_mass_plot(
        eff_mass,
        get_output_filename(
            output_filename_prefix,
            "effmass",
            f"{valence_mass}_spin12",
        ),
        ymin=eff_mass_plot_ymin,
        ymax=eff_mass_plot_ymax,
    )

    do_correlator_plot(
        correlator,
        get_output_filename(output_filename_prefix, "correlator", "spin12"),
        r"\breve{g}",
    )

    if not (plateau_start and plateau_end):
        raise Incomplete(
            "Effective mass plot has been generated. "
            "Now specify the start and end of the plateau to "
            "perform the fit."
        )

    result = fit_single_correlator(correlator, [plateau_start, plateau_end])

    do_correlator_plot(
        correlator,
        get_output_filename(
            output_filename_prefix,
            "centrally_fitted_correlator",
            channel=f"{valence_mass}_spin12",
            tstart=plateau_start,
            tend=plateau_end,
        ),
        r"\breve{g}",
        fit_function=get_fit_form(NT, "v"),
        fit_params={"params": [v.value for v in result], "NT": NT},
        fit_legend="Fit",
        t_lowerbound=plateau_start - 3.5,
        t_upperbound=plateau_end + 0.5,
        corr_upperbound=correlator_upperbound,
        corr_lowerbound=correlator_lowerbound,
    )

    do_eff_mass_plot(
        eff_mass,
        get_output_filename(
            output_filename_prefix,
            "effmass_withfit",
            channel=f"{valence_mass}_spin12",
            tstart=plateau_start,
            tend=plateau_end,
        ),
        ymin=eff_mass_plot_ymin,
        ymax=eff_mass_plot_ymax,
        m=result[0],
        tmin=plateau_start - 0.5,
        tmax=plateau_end + 0.5,
    )

    return result[0]


def plot_measure_and_save_spin12(
    simulation_descriptor,
    correlator_directory,
    output_filename_prefix=None,
    spin12_parameters=None,
    parameter_date=None,
    force=False,
):
    if not spin12_parameters:
        spin12_parameters = {}

    if not output_filename_prefix:
        output_filename_prefix = correlator_filename + "_"

    need_to_run = False
    if force:
        need_to_run = True
    else:
        if not measurement_is_up_to_date(
            simulation_descriptor,
            "spin12_mass",
            valence_mass=simulation_descriptor["m"],
            compare_glob=f"{correlator_directory}/out_corr_spin12*",
        ):
            need_to_run = True
        if parameter_date and not measurement_is_up_to_date(
            simulation_descriptor,
            "spin12_mass",
            valence_mass=simulation_descriptor["m"],
            compare_date=parameter_date,
        ):
            need_to_run = True
    if not need_to_run:
        return

    valence_mass = simulation_descriptor["m"]
    fit_result = process_correlator(
        correlator_directory,
        simulation_descriptor["T"],
        valence_mass,
        output_filename_prefix=output_filename_prefix,
        **spin12_parameters,
    )

    add_measurement(
        simulation_descriptor,
        "spin12_mass",
        fit_result,
        valence_mass=valence_mass,
    )

    return fit_result
