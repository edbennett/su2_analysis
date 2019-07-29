from numpy import argmax
from matplotlib.pyplot import subplots

from .bootstrap import basic_bootstrap, bootstrap_1d
from .data import get_flows_from_raw
from .db import measurement_is_up_to_date, add_measurement
from .data import get_filename


def ensemble_sqrt_8t0(times, Es, E0, ax=None, plot_label=None):
    h = times[1] - times[0]
    t2E = times ** 2 * Es
    if ax:
        t2E_avg, t2E_err = bootstrap_1d(t2E)

        ax.errorbar(
            times, t2E_avg, yerr=t2E_err,
            fmt='.',
            label=plot_label
        )
    positions = argmax(t2E > E0, axis=1)

    T_positions_minus_one = t2E[tuple(zip(*enumerate(positions - 1)))]
    T_positions = t2E[tuple(zip(*enumerate(positions)))]

    t0 = times[positions] + h * (
        (E0 - T_positions_minus_one) /
        (T_positions - T_positions_minus_one)
    )
    return (8 * t0) ** 0.5


def measure_w0(filename, E0, ax=None):
    '''Reads flows from`filename`, and finds the value t where  E(t) == `E0`.
    Plots flows on `ax` if it is given.

    Returns (s8t0p, s8t0p_error), (s8t0c, s8t0c_error)'''

    trajectories, times, Eps, Ecs, _ = get_flows_from_raw(filename)
    s8t0ps = ensemble_sqrt_8t0(times, Eps, E0, ax=ax,
                               plot_label=r'$\sqrt{8t_0^p}$')
    s8t0cs = ensemble_sqrt_8t0(times, Ecs, E0, ax=ax,
                               plot_label=r'$\sqrt{8t_0^c}$')

    s8t0p = basic_bootstrap(s8t0ps)
    s8t0c = basic_bootstrap(s8t0cs)

    return s8t0p, s8t0c


def plot_measure_and_save_sqrt_8t0(E0,
                                   simulation_descriptor=None,
                                   filename_formatter=None,
                                   filename=None,
                                   plot_filename_formatter=None,
                                   plot_filename=None,
                                   force=False):
    '''Measure sqrt{8t0} via plaquette and clover for a given simulation
    described by `simulation_descriptor`, with gradient flow output file in
    either `filename_formatter(simulation_descriptor)` or `filename`.
    Use `E0` as reference scale.
    Only calculate if the result in the database is older than the flows
    file, or `force=True`. Plot flows in the file
    `plot_filename_formatter(simulation_descriptor)`.
    Save results to the database.'''

    filename = get_filename(simulation_descriptor,
                            filename_formatter, filename)
    plot_filename = get_filename(simulation_descriptor,
                                 plot_filename_formatter,
                                 plot_filename,
                                 optional=True)

    if (simulation_descriptor
        and not force
        and (measurement_is_up_to_date(simulation_descriptor, 's8t0p',
                                       compare_file=filename,
                                       free_parameter=E0)
             and measurement_is_up_to_date(simulation_descriptor, 's8t0c',
                                           compare_file=filename,
                                           free_parameter=E0))):
        # Already up to date
        return

    if plot_filename:
        fig, ax = subplots()
    else:
        fig, ax = None, None

    s8t0p, s8t0c = measure_w0(filename, E0, ax=ax)

    if simulation_descriptor:
        add_measurement(simulation_descriptor, 's8t0p', *s8t0p,
                        free_parameter=E0)
        add_measurement(simulation_descriptor, 's8t0c', *s8t0c,
                        free_parameter=E0)

    if plot_filename:
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$t^2 E$')
        ax.legend(loc=0, frameon=False)
        ax.axhline(E0, dashes=(1, 1))
        fig.tight_layout()
        fig.savefig(plot_filename)

    return s8t0p, s8t0c
