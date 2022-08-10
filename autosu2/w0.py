import warnings

from numpy import argmax
from matplotlib.pyplot import subplots, close
from argparse import ArgumentParser

from .bootstrap import basic_bootstrap, sample_bootstrap_1d, bootstrap_1d
from .data import get_flows_from_raw
from .db import (
    measurement_is_up_to_date, add_measurement, get_measurement,
    measurement_exists
)
from .data import get_filename


DEFAULT_W0 = 0.2


def ensemble_w0(times, Es, W0, ax=None, plot_label=None):
    h = times[1] - times[0]
    t2E = times ** 2 * Es
    tdt2Edt = times[1:-1] * (t2E[:, 2:] - t2E[:, :-2]) / (2 * h)
    tdt2Edt_samples = sample_bootstrap_1d(tdt2Edt)

    if ax:
        ax.errorbar(
            times[1:-1],
            tdt2Edt_samples.mean(axis=0),
            yerr=tdt2Edt_samples.std(axis=0),
            fmt='.',
            label=plot_label
        )

    positions = argmax(tdt2Edt_samples > W0, axis=1)
    W_positions_minus_one = tdt2Edt_samples[tuple(zip(*enumerate(positions - 1)))]
    W_positions = tdt2Edt_samples[tuple(zip(*enumerate(positions)))]
    w0_squared = times[positions] + h * (
        (W0 - W_positions_minus_one) /
        (W_positions - W_positions_minus_one)
    )
    if min(positions) == 0:
        bad_ratio = sum(positions == 0) / len(positions)
        warnings.warn(f"w0: {bad_ratio:%} of samples do not reach W0 = {W0}")
        w0_squared = w0_squared[positions > 0]

    return w0_squared ** 0.5


def measure_w0(filename, W0, bin_size=1, ax=None):
    '''Reads flows from`filename`, and finds the value t where  W(t) == W0.
    Plots flows on `ax` if it is given.

    Returns (w0p, w0p_error), (w0c, w0c_error)'''

    trajectories, times, Eps, Ecs, _ = get_flows_from_raw(filename, bin_size)
    bs_w0ps = ensemble_w0(times, Eps, W0, ax=ax, plot_label=r'$w_0^p$')
    bs_w0cs = ensemble_w0(times, Ecs, W0, ax=ax, plot_label=r'$w_0^c$')

    w0p = bs_w0ps.mean(), bs_w0ps.std()
    w0c = bs_w0cs.mean(), bs_w0cs.std()

    return w0p, w0c


def plot_measure_and_save_w0(W0,
                             simulation_descriptor=None,
                             filename_formatter=None,
                             filename=None,
                             plot_filename_formatter=None,
                             plot_filename=None,
                             force=False,
                             bin_size=1):
    '''Measure w0 via plaquette and clover for a given simulation described
    by `simulation_descriptor`, with gradient flow output file in either
    `filename_formatter(simulation_descriptor)` or `filename`.
    Use `W0` as reference scale.
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
        and (measurement_is_up_to_date(simulation_descriptor, 'w0p',
                                       compare_file=filename,
                                       free_parameter=W0)
             and measurement_is_up_to_date(simulation_descriptor, 'w0c',
                                           compare_file=filename,
                                           free_parameter=W0))):
        # Already up to date
        return

    if simulation_descriptor and measurement_exists(
            simulation_descriptor, 'Q_tau_exp'
    ):
        bin_size = int(
            get_measurement(simulation_descriptor, 'Q_tau_exp').value
        ) + 1

    if plot_filename:
        fig, ax = subplots()
    else:
        fig, ax = None, None

    w0p, w0c = measure_w0(filename, W0, bin_size=bin_size, ax=ax)

    if simulation_descriptor:
        add_measurement(simulation_descriptor, 'w0p', *w0p, free_parameter=W0)
        add_measurement(simulation_descriptor, 'w0c', *w0c, free_parameter=W0)

    if plot_filename:
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$t \frac{\mathrm{d}(t^2 E)}{\mathrm{d}t}$')
        ax.legend(loc=0, frameon=False)
        ax.axhline(W0, dashes=(1, 1))
        fig.tight_layout()
        fig.savefig(plot_filename)
        close(fig)

    return w0p, w0c


def main():
    parser = ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--W0', default=DEFAULT_W0, type=float)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--bin_size', default=1, type=int)
    args = parser.parse_args()

    if not args.output_filename_prefix:
        args.output_filename_prefix = args.filename + '_'

    if args.plot:
        plot_filename = args.output_filename_prefix + 'w0.pdf'
    else:
        plot_filename = None

    result = plot_measure_and_save_w0(
        W0=args.W0,
        filename=args.filename,
        plot_filename=plot_filename,
        bin_size=args.bin_size
    )
    if result:
        w0p, w0c = result

        if not args.silent:
            print(f"w0p: {w0p[0]} ± {w0p[1]}")
            print(f"w0c: {w0c[0]} ± {w0c[1]}")

    elif not args.silent:
        print("No results returned.")

    # write_results(
    #     filename=get_output_filename(
    #         args.output_filename_prefix, 'w0', filetype='dat'
    #     ),
    #     headers=('w0p', 'w0p_error', 'w0c', 'w0c_error'),
    #     values_set=(w0p, w0c)
    # )


if __name__ == '__main__':
    main()
