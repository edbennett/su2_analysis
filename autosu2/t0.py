from matplotlib.pyplot import subplots, close

from flow_analysis.measurements.scales import compute_t2E_t, measure_sqrt_8t0
from flow_analysis.readers import readers

from .db import (
    measurement_is_up_to_date,
    add_measurement,
)
from .data import get_filename

DEFAULT_E0 = 0.2


def plot_measure_and_save_sqrt_8t0(
    E0,
    simulation_descriptor=None,
    filename_formatter=None,
    filename=None,
    plot_filename_formatter=None,
    plot_filename=None,
    force=False,
    reader="hirep",
):
    """Measure sqrt{8t0} via plaquette and clover for a given simulation
    described by `simulation_descriptor`, with gradient flow output file in
    either `filename_formatter(simulation_descriptor)` or `filename`.
    Use `E0` as reference scale.
    Only calculate if the result in the database is older than the flows
    file, or `force=True`. Plot flows in the file
    `plot_filename_formatter(simulation_descriptor)`.
    Save results to the database."""

    filename = get_filename(simulation_descriptor, filename_formatter, filename)
    plot_filename = get_filename(
        simulation_descriptor, plot_filename_formatter, plot_filename, optional=True
    )

    if (
        simulation_descriptor
        and not force
        and (
            measurement_is_up_to_date(
                simulation_descriptor, "s8t0p", compare_file=filename, free_parameter=E0
            )
            and measurement_is_up_to_date(
                simulation_descriptor, "s8t0c", compare_file=filename, free_parameter=E0
            )
        )
    ):
        # Already up to date
        return

    results = []
    flows = readers[reader](filename)

    if plot_filename:
        fig, ax = subplots()
    else:
        fig, ax = None, None

    for operator, suffix in ("plaq", "p"), ("sym", "c"):
        try:
            s8t0 = measure_sqrt_8t0(flows, E0, operator=operator)
        except ValueError:
            s8t0 = None

        results.append(s8t0)

        if s8t0 and simulation_descriptor:
            add_measurement(
                simulation_descriptor, f"s8t0{suffix}", s8t0, free_parameter=E0
            )

        if plot_filename:
            t2E_mean, t2E_error = compute_t2E_t(flows, operator=operator)
            ax.errorbar(
                flows.times,
                t2E_mean,
                yerr=t2E_error,
                fmt=".",
                label=f"$\\sqrt{{8t_0^{suffix}}}$",
            )

    if plot_filename:
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$t^2 E$")
        ax.legend(loc=0, frameon=False)
        ax.axhline(E0, dashes=(1, 1))
        fig.tight_layout()
        fig.savefig(plot_filename)
        close(fig)

    return results
