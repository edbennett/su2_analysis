#!/usr/bin/env python3

from collections.abc import Sequence
import logging

import matplotlib.pyplot as plt
import numpy as np

from glue_analysis.readers import readers
from meson_analysis.fit_forms import get_fit_form

from .db import add_measurement, get_measurement_as_ufloat, measurement_is_up_to_date
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
            if not measurement_is_up_to_date(
                simulation_descriptor, db_column, compare_file=filename
            ):
                return True

        for compare_date in compare_dates if compare_dates else []:
            if not compare_date:
                continue
            if not measurement_is_up_to_date(
                simulation_descriptor, db_column, compare_date=compare_date
            ):
                return True

    return False


def get_ground_state_correlators(raw_correlator, channel, subtract=True):
    correlator = raw_correlator.get_pyerrors(subtract=subtract)
    correlator.gamma_method()
    return [correlator.Eigenvalue(t0=0, state=state) for state in STATES[channel]]


def plot_eff_masses(
    ground_state,
    output_filename,
    plateaux,
    extra_states=[],
    title=None,
    result=None,
):
    states_to_plot = {
        "Combined ground state": ground_state,
        **{f"State {idx}": state for idx, state in enumerate(extra_states)},
    }

    fig, ax = plt.subplots()

    for idx, plateau in enumerate(plateaux):
        if not plateau:
            continue
        for plateau_end, offset in zip(plateau, (-0.5, 0.5)):
            if plateau_end is not None:
                shift = 0.05
                ax.axvline(
                    plateau_end + offset + idx * shift,
                    lw=2,
                    color=f"C{idx + 1}",
                    dashes=(2, 3),
                )

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
            ax.axhline(
                result.value + sign * result.dvalue, color="black", dashes=[2, 3]
            )

    if title is not None:
        ax.set_title(title)
    if extra_states or result:
        ax.legend(loc="best")

    fig.savefig(output_filename)
    plt.close(fig)


def string_tension_from_torelon(torelon_mass, torelon_length):
    string_tension = (
        np.pi + np.sqrt(np.pi**2 + 9 * torelon_mass**2 * torelon_length**2)
    ) / (3 * torelon_length**2)
    string_tension.gamma_method()
    return string_tension


def weighted_mean(observations, error_attr, callback_method=None):
    numerator_elements = (obs / getattr(obs, error_attr) ** 2 for obs in observations)
    denominator_elements = (1 / getattr(obs, error_attr) ** 2 for obs in observations)
    result = sum(numerator_elements) / sum(denominator_elements)
    if callback_method:
        getattr(result, callback_method)()
    return result


def weighted_mean_pyerrors(observations):
    return weighted_mean(observations, "dvalue", "gamma_method")


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

    if (
        not need_to_run(
            simulation_descriptor,
            [correlator_filename, vev_filename],
            [parameter_date],
            [f"{channel_name}_mass"],
        )
        and not force
    ):
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

    if "plateau_start" in glue_parameters and "plateau_end" in glue_parameters:
        plateaux = [[glue_parameters["plateau_start"], glue_parameters["plateau_end"]]]
    elif isinstance(glue_parameters, Sequence) and not isinstance(glue_parameters, str):
        plateaux = [
            [plateau["plateau_start"], plateau["plateau_end"]] if plateau else None
            for plateau in glue_parameters
        ]
    else:
        return

    ground_states = get_ground_state_correlators(raw_correlator, channel=channel_name)
    for state in ground_states:
        state.gamma_method()
    combined_ground_state = sum(ground_states) / len(ground_states)
    combined_ground_state.gamma_method()

    plot_eff_masses(
        combined_ground_state,
        f"{output_filename_prefix}effmass_{channel_name}.pdf",
        plateaux,
        extra_states=ground_states if len(ground_states) > 1 else [],
        title=f"{simulation_descriptor['label']}, {channel_name}",
    )

    if all([not all(plateau) for plateau in plateaux if plateau is not None]):
        return

    try:
        results = [
            state.fit(
                get_fit_form(simulation_descriptor["T"], "v"),
                plateau,
                silent=True,
                method="migrad",
            )
            for state, plateau in zip(ground_states, plateaux)
            if plateau
        ]
        for result in results:
            result.gamma_method()

        fit_mass = weighted_mean_pyerrors(
            [result.fit_parameters[0] for result in results]
        )
    except ValueError as ex:
        message = f"Error in fit_glue: {ex}"
        logging.warning(message)
        return

    plateaux_descriptor = "_".join(
        ["-".join(map(str, plateau)) if plateau else "--" for plateau in plateaux]
    )
    plot_eff_masses(
        combined_ground_state,
        f"{output_filename_prefix}effmass_{channel_name}_withfit_{plateaux_descriptor}.pdf",
        plateaux,
        extra_states=ground_states if len(ground_states) > 1 else [],
        title=f"{simulation_descriptor['label']}, {channel_name}",
        result=fit_mass,
    )

    add_measurement(
        simulation_descriptor,
        f"{channel_name}_mass",
        fit_mass,
    )

    if channel_name == "torelon":
        string_tension = string_tension_from_torelon(
            result.fit_parameters[0], simulation_descriptor["L"]
        )
        add_measurement(simulation_descriptor, "string_tension", string_tension)

        sqrtsigma = string_tension**0.5
        sqrtsigma.gamma_method()
        add_measurement(simulation_descriptor, "sqrtsigma", sqrtsigma)

    return fit_mass


def select_2plusplus_state(simulation_descriptor, Epp_params, T2pp_params):
    try:
        Epp = get_measurement_as_ufloat(simulation_descriptor, "E++_mass")
    except KeyError:
        Epp = None

    try:
        T2pp = get_measurement_as_ufloat(simulation_descriptor, "T2++_mass")
    except KeyError:
        T2pp = None

    use = lambda mass: add_measurement(simulation_descriptor, "2++_mass", mass)

    if Epp is None and T2pp is None:
        purge_measurement(simulation_descriptor, "2++_mass")

    elif Epp is None:
        use(T2pp)

    elif T2pp is None:
        use(Epp)

    elif Epp_params.get("use") and T2pp_params.get("use"):
        use(weighted_mean([T2pp, Epp], "std_dev"))

    elif Epp_params.get("use"):
        use(Epp)

    elif T2pp_params.get("use"):
        use(T2pp)

    else:
        purge_measurement(simulation_descriptor, "2++_mass")

    try:
        return get_measurement_as_ufloat(simulation_descriptor, "2++_mass")
    except KeyError:
        return
