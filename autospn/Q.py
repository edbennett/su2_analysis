import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from collections import Counter
import argparse

from .bootstrap import basic_bootstrap, bootstrap_susceptibility
from .data import get_flows_from_raw
from .data import file_is_up_to_date
from .db import measurement_is_up_to_date, add_measurement
from .plots import set_plot_defaults


def gaussian(x, A, x0, sigma):
    '''Returns A e^(-(x - x0)^2 / 2 sigma^2)'''

    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def plot_history_and_histogram(trajectories, Qs, output_file=None,
                               title=True, extra_title='', legend=False):
    '''Histograms, amd plots to the screen or a PDF if specified,
    the topological charges in Qs, with trajectory numbers provided in
    `trajectories`.'''

    set_plot_defaults()
    f, (history, histogram) = plt.subplots(
        1, 2, sharey=True,
        gridspec_kw={'width_ratios': [3, 1]}, figsize=(5.5, 2.5)
    )

    Q_bins = Counter(Qs.round())
    history.step(trajectories, Qs)

    history.set_xlabel('Trajectory')
    history.set_ylabel('$Q$')
    if legend:
        history.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                       ncol=5, mode="expand", borderaxespad=0.,
                       frameon=False)

    # One extra to include a zero
    range_min = min(min(Q_bins), -max(Q_bins)) - 1
    # Add one to be inclusive
    range_max = -range_min + 1

    history.set_ylim([range_min - 0.5, range_max - 0.5])

    Q_range = np.arange(range_min, range_max)
    Q_counts = [Q_bins[Q] for Q in Q_range]

    histogram.step(Q_counts, Q_range - 0.5, label="Histogram")

    (A, Q0, sigma), pcov = curve_fit(gaussian, Q_range, Q_counts)
    smooth_Q_range = np.linspace(range_min - 0.5, range_max - 0.5, 1000)
    histogram.plot(gaussian(smooth_Q_range, A, Q0, sigma),
                   smooth_Q_range, label="Fit")
#    histogram.legend(loc=0, frameon=False)
    histogram.set_xlabel("Count")

    f.tight_layout()

    if title:
        f.suptitle(('' if extra_title is None else extra_title) +
                   r" $Q_0 = {:.2f} \pm {:.2f}$; $\sigma = {:.2f} \pm {:.2f}$"
                   .format(Q0, np.sqrt(pcov[1][1]),
                           abs(sigma), np.sqrt(pcov[2][2])))
        f.subplots_adjust(top=0.8 if legend else 0.9)
    if output_file is None:
        f.show()
    else:
        f.savefig(output_file)

    return (
        # r"$Q_0 = {:.2f} \pm {:.2f}$; $\sigma = {:.2f} \pm {:.2f}$".format(
        (Q0, np.sqrt(pcov[1][1])), (abs(sigma), np.sqrt(pcov[2][2]))
    )

    return (Q0, np.sqrt(pcov[1][1])), (sigma, np.sqrt(pcov[2][2]))


def topological_charge_susceptibility(Qs, V):
    Q0 = basic_bootstrap(Qs)
    chi_top = tuple(value / V for value in bootstrap_susceptibility(Qs))

    return Q0, chi_top


def plot_measure_and_save_Q(flows_file, simulation_descriptor=None,
                            output_file=None):
    '''Reads in flows_file in HiRep~MILC format, calculates average
    Q and topological susceptibility, plots and histograms, saves
    to the database.'''

    if file_is_up_to_date(output_file, compare_file=flows_file):
        if simulation_descriptor:
            if (measurement_is_up_to_date(
                    simulation_descriptor, 'fitted_Q0', compare_file=flows_file
            ) and measurement_is_up_to_date(
                    simulation_descriptor, 'Q_width', compare_file=flows_file
            )):
                # File and database both up-to-date
                do_fit_and_plot = False
            else:
                # Database is out of date
                do_fit_and_plot = True
        else:
            # File is up to date, no database
            do_fit_and_plot = False
    else:
        # File is out of date
        do_fit_and_plot = True

    if simulation_descriptor:
        if measurement_is_up_to_date(
                simulation_descriptor, 'Q0', compare_file=flows_file
        ) and measurement_is_up_to_date(
            simulation_descriptor, 'chi_top', compare_file=flows_file
        ):
            do_bootstrap = False
        else:
            do_bootstrap = True
    else:
        do_bootstrap = True

    result = {}

    if do_fit_and_plot or do_bootstrap:
        trajectories, *_, Qs = get_flows_from_raw(flows_file)

    if do_fit_and_plot:
        fitted_Q0, Q_width = plot_history_and_histogram(
            trajectories, Qs, output_file=output_file, title=False
        )
        if simulation_descriptor:
            add_measurement(simulation_descriptor, 'fitted_Q0', *fitted_Q0)
            add_measurement(simulation_descriptor, 'Q_width', *Q_width)
        result['fitted_Q0'] = fitted_Q0
        result['Q_width'] = Q_width

    if do_bootstrap:
        Q0, chi_top = topological_charge_susceptibility(
            Qs, simulation_descriptor['T'] * simulation_descriptor['L'] ** 3
        )
        add_measurement(simulation_descriptor, 'Q0', *Q0)
        add_measurement(simulation_descriptor, 'chi_top', *chi_top)
        result['Q0'] = Q0
        result['chi_top'] = chi_top
    return result


def main():
    '''Runs the program'''

    parser = argparse.ArgumentParser(description="Plot the topological "
                                     "charge history and histogram")
    parser.add_argument("flows_file", help="A Q history in HiRep~MILC format")
    parser.add_argument("--output-file", default=None,
                        help="PDF file to output to. "
                        "If omitted, outputs to the screen")
    parser.add_argument("--no-title", action="store_false", dest="title",
                        help="Disable the plot title.")
    parser.add_argument("--with-legend", action="store_true", dest="legend",
                        help="Enable the legend on the history plot.")
    parser.add_argument("--extra-title",
                        help="Extra text to add to the title")
    args = parser.parse_args()

    trajectories, *_, Qs = get_flows_from_raw(args.flows_file)

    plot_history_and_histogram(
        trajectories=trajectories, Qs=Qs,
        **{k: vars(args)[k]
           for k in ('output-file', 'no-title', 'with-legend', 'extra-title')}
    )


if __name__ == '__main__':
    main()
