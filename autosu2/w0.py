import warnings

from numpy import argmax
from matplotlib.pyplot import subplots, close
from argparse import ArgumentParser

from flow_analysis.measurements.scales import compute_wt_t, measure_w0
from flow_analysis.readers import readers

from .bootstrap import basic_bootstrap, sample_bootstrap_1d, bootstrap_1d
from .data import get_flows_from_raw
from .db import (
    measurement_is_up_to_date, add_measurement, get_measurement,
    measurement_exists
)
from .data import get_filename


DEFAULT_W0 = 0.2


def plot_measure_and_save_w0(W0,
                             simulation_descriptor=None,
                             filename_formatter=None,
                             filename=None,
                             plot_filename_formatter=None,
                             plot_filename=None,
                             force=False,
                             reader="hirep"):
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

    results = []
    flows = readers[reader](filename)

    if plot_filename:
        fig, ax = subplots()

    for operator, suffix in ("plaq", "p"), ("sym", "c"):
        try:
            w0 = measure_w0(flows, W0, operator=operator)
        except ValueError:
            w0 = None

        results.append(w0)

        if w0 and simulation_descriptor:
            add_measurement(simulation_descriptor, f'w0{suffix}', w0, free_parameter=W0)

        if plot_filename:
            w_mean, w_error = compute_wt_t(flows, W0)
            ax.errorbar(flows.times[1:-1], w_mean, yerr=w_error, fmt=".", label=f"$w_0^{suffix}$")

    if plot_filename:
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$t \frac{\mathrm{d}(t^2 E)}{\mathrm{d}t}$')
        ax.legend(loc=0, frameon=False)
        ax.axhline(W0, dashes=(1, 1))
        fig.tight_layout()
        fig.savefig(plot_filename)
        close(fig)

    return results


def main():
    parser = ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--W0', default=DEFAULT_W0, type=float)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--silent', action='store_true')
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
    )
    if result:
        w0p, w0c = result

        if not args.silent:
            print(f"w0p: {w0p}")
            print(f"w0c: {w0c}")

    elif not args.silent:
        print("No results returned.")


if __name__ == '__main__':
    main()
