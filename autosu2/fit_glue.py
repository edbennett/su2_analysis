#!/usr/bin/env python3

import logging

import matplotlib.pyplot as plt
import numpy as np

from glue_analysis.readers import readers
from meson_analysis.fit_forms import get_fit_form

from .db import add_measurement, measurement_is_up_to_date
from .plots import set_plot_defaults


STATES = {
    "torelon": (0,),
    "A1++": (0,),
    "E++": (0, 1),
    "T2++": (0, 1, 2),
}

def need_to_run(simulation_descriptor, filenames, compare_dates, db_columns):
    for db_column in db_columns:
        for filename in filenames if filenames else []:
            if not filename:
                continue
            if not measurement_is_up_to_date(simulation_descriptor, db_column, compare_file=filename):
                return True

        for compare_date in compare_dates if compare_dates else []:
            if not compare_date:
                continue
            if not measurement_is_up_to_date(simulation_descriptor, db_column, compare_date=compare_date):
                return True

    return False


def get_ground_state_correlators(raw_correlator, channel, subtract=True):
    correlator = raw_correlator.get_pyerrors(subtract=subtract)
    correlator.gamma_method()
    return [correlator.Eigenvalue(t0=0, state=state) for state in STATES[channel]]


def plot_eff_masses(ground_state, output_filename, glue_parameters, extra_states=[], title=None, result=None):
    states_to_plot = {
        "Combined ground state": ground_state,
        **{
            f"State {idx}": state for idx, state in enumerate(extra_states)
        }
    }

    fig, ax = plt.subplots()
    for position, offset in ("plateau_start", -0.5), ("plateau_end", 0.5):
        ax.axvline(glue_parameters.get(position, np.nan) + offset, lw=2, color="orange")

    for idx, (label, state) in enumerate(states_to_plot.items()):
        m_eff = state.m_eff(variant="log")
        m_eff.gamma_method()

        timeslice, m_eff_value, m_eff_error = m_eff.plottable()
        ax.errorbar(
            [t + idx * 0.1 for t in timeslice],
            m_eff_value,
            yerr=m_eff_error,
            ls="none",
            marker="o",
            label=label,
        )
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$m_{\mathrm{eff}}$")

    if result:
        ax.axhline(result.value, label="Fit result", color="black")
        for sign in 1, -1:
            ax.axhline(result.value + sign * result.dvalue, color="black", dashes=[2, 3])

    if title is not None:
        ax.set_title(title)
    if extra_states or result:
        ax.legend(loc="best")

    fig.savefig(output_filename)


def string_tension_from_torelon(torelon_mass, torelon_length):
    string_tension = (
        (np.pi + np.sqrt(np.pi ** 2 + 9 * torelon_mass ** 2 * torelon_length ** 2))
        / (3 * torelon_length ** 2)
    )
    string_tension.gamma_method()
    return string_tension


def plot_measure_and_save_glueballs(
        simulation_descriptor,
        correlator_filename,
        vev_filename,
        channel_name,
        num_configs,
        output_filename_prefix=None,
        glue_parameters=None,
        parameter_date=None,
        force=False,
        reader="fortran",
):
    if not glue_parameters:
        glue_parameters = {}

    if not output_filename_prefix:
        output_filename_prefix = correlator_filename + "_"

    if not need_to_run(simulation_descriptor, [correlator_filename, vev_filename], [parameter_date], [f"{channel_name}_mass"]) and not force:
        return

    set_plot_defaults(markersize=2)

    raw_correlator = readers[reader](
        corr_filename=correlator_filename,
        vev_filename=vev_filename,
        channel=channel_name,
        metadata={
            "NT": simulation_descriptor["T"],
            "num_configs": num_configs,
        },
    )
    ground_states = get_ground_state_correlators(raw_correlator, channel=channel_name)
    combined_ground_state = sum(ground_states) / len(ground_states)
    combined_ground_state.gamma_method()

    plot_eff_masses(
        combined_ground_state,
        f"{output_filename_prefix}effmass_{channel_name}.pdf",
        glue_parameters,
        extra_states=ground_states if len(ground_states) > 1 else [],
        title=f"{simulation_descriptor['label']}, {channel_name}"
    )

    if not (
        (plateau_start := glue_parameters.get("plateau_start")) is not None
        and (plateau_end := glue_parameters.get("plateau_end")) is not None
    ):
        return

    try:
        result = combined_ground_state.fit(
            get_fit_form(simulation_descriptor["T"], "v"),
            [plateau_start, plateau_end],
            silent=True,
            method="migrad",
        )
    except ValueError:# Exception as ex:
        logging.warning(ex)
        return

    result.gamma_method()
    plot_eff_masses(
        combined_ground_state,
        f"{output_filename_prefix}effmass_{channel_name}_withfit_{plateau_start}_{plateau_end}.pdf",
        glue_parameters,
        extra_states=ground_states if len(ground_states) > 1 else [],
        title=f"{simulation_descriptor['label']}, {channel_name}",
        result=result.fit_parameters[0],
    )

    add_measurement(
        simulation_descriptor,
        f"{channel_name}_mass",
        result.fit_parameters[0],
    )

    if channel_name == "torelon":
        string_tension = string_tension_from_torelon(result.fit_parameters[0], simulation_descriptor["L"])
        add_measurement(simulation_descriptor, f"string_tension", string_tension)

        sqrtsigma = string_tension ** 0.5
        sqrtsigma.gamma_method()
        add_measurement(simulation_descriptor, f"sqrtsigma", sqrtsigma)

    return result.fit_parameters[0]
