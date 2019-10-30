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
    '''
    Gaussian fit form
    Returns A e^(-(x - x0)^2 / 2 sigma^2)
    '''

    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def exp_decay(x, A, tau_exp):
    '''
    The fit form of an exponential decay.
    Returns A e^(-x / tau_exp)
    '''

    return A * np.exp(-x / tau_exp)


def autocorr(series, cutoff=None):
    '''
    Calculate the autocorrelation function of the `series`.
    If `cutoff` isn't specified, it defaults to `len(series) // 2`.
    '''

    if not cutoff:
        cutoff = len(series) // 2
    acf = np.zeros(cutoff - 1)
    series -= np.mean(series)
    acf[0] = 1
    for i in range(1, cutoff - 1):
        acf[i] = np.mean(series[:-i] * series[i:]) / np.var(series)
    return acf


def analyse_autocorrelation(series, filename, fit_range=10):
    '''
    Calculates and plots the autocorrelation function of `series`.
    Outputs to `filename`, or to screen if this is not specified.
    Fits the exponential autocorrelation time of the first `fit_range` points
    of `series` and returns it with its uncertainty.
    '''

    set_plot_defaults()
    f, ax = plt.subplots()

    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$C(\tau)$')

    acf = autocorr(series)
    ax.scatter(np.arange(len(acf)), acf)

    fit_result = curve_fit(exp_decay,
                           np.arange(fit_range),
                           acf[:fit_range])
    acf_domain = np.linspace(0, len(acf), 1000)
    ax.plot(acf_domain, exp_decay(acf_domain, *fit_result[0]))

    if filename:
        f.savefig(filename)
        plt.close(f)
    else:
        plt.show()

    tau_exp, tau_exp_error = fit_result[0][1], fit_result[1][1][1] ** 0.5

    if tau_exp < 1 and tau_exp_error > 10 * tau_exp:
        if np.std(acf[1:fit_range]) > np.mean(acf[1:fit_range]):
            # Autocorrelation time is much less than 1
            # Not enough resolution for fitter to determine precise location
            tau_exp = 0
            tau_exp_error = 0.5

    return tau_exp, tau_exp_error


def plot_history_and_histogram(trajectories, Qs, output_file=None,
                               title=True, extra_title='', legend=False,
                               history_ax=None, histogram_ax=None):
    '''Histograms, amd plots to the screen or a PDF if specified,
    the topological charges in Qs, with trajectory numbers provided in
    `trajectories`.'''

    assert (
        (not history_ax and not histogram_ax) or (history_ax and histogram_ax)
    )
    assert not (output_file and history_ax)

    if not history_ax:
        set_plot_defaults()
        f, (history_ax, histogram_ax) = plt.subplots(
            1, 2, sharey=True,
            gridspec_kw={'width_ratios': [3, 1]}, figsize=(5.5, 2.5)
        )
        if legend:
            history_ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                              ncol=5, mode="expand", borderaxespad=0.,
                              frameon=False)

    history_ax.set_xlabel('Trajectory')
    history_ax.set_ylabel('$Q$')

    Q_bins = Counter(Qs.round())
    history_ax.step(trajectories, Qs)

    # One extra to include a zero
    range_min = min(min(Q_bins), -max(Q_bins)) - 1
    # Add one to be inclusive
    range_max = -range_min + 1

    history_ax.set_ylim([range_min - 0.5, range_max - 0.5])

    Q_range = np.arange(range_min, range_max)

    Q_counts = [Q_bins[Q] for Q in Q_range]

    histogram_ax.step(Q_counts, Q_range - 0.5, label="Histogram")

    (A, Q0, sigma), pcov = curve_fit(gaussian, Q_range, Q_counts)
    smooth_Q_range = np.linspace(range_min - 0.5, range_max - 0.5, 1000)
    histogram_ax.plot(gaussian(smooth_Q_range, A, Q0, sigma),
                      smooth_Q_range, label="Fit")
#    histogram.legend(loc=0, frameon=False)

    histogram_ax.set_xlabel("Count")

    if not history_ax:
        f.tight_layout()

        if title:
            f.suptitle(
                ('' if extra_title is None else extra_title) +
                r" $Q_0 = {:.2f} \pm {:.2f}$; $\sigma = {:.2f} \pm {:.2f}$"
                .format(Q0, np.sqrt(pcov[1][1]),
                        abs(sigma), np.sqrt(pcov[2][2]))
            )
            f.subplots_adjust(top=0.8 if legend else 0.9)
    if output_file is None:
        plt.show()
    else:
        f.savefig(output_file)
    plt.close(f)

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
                            output_file_history=None,
                            output_file_autocorr=None):
    '''Reads in flows_file in HiRep~MILC format, calculates average
    Q and topological susceptibility, plots and histograms, saves
    to the database.'''

    if file_is_up_to_date(output_file_history, compare_file=flows_file):
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
        tau_exp = analyse_autocorrelation(Qs, output_file_autocorr)
        if simulation_descriptor:
            add_measurement(simulation_descriptor, 'Q_tau_exp', *tau_exp)

    if do_fit_and_plot:
        fitted_Q0, Q_width = plot_history_and_histogram(
            trajectories, Qs, output_file=output_file_history, title=False
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
    parser.add_argument("--output_file", default=None,
                        help="PDF file to output history and histogram to. "
                        "If omitted, outputs to the screen")
    parser.add_argument("--output_file_autocorr", default=None,
                        help="PDF file to output autocorrelation plot to. "
                        "If omitted, outputs to the screen")
    parser.add_argument("--no_title", action="store_false", dest="title",
                        help="Disable the plot title.")
    parser.add_argument("--with_legend", action="store_true", dest="legend",
                        help="Enable the legend on the history plot.")
    parser.add_argument("--extra_title",
                        help="Extra text to add to the title")
    args = parser.parse_args()

    trajectories, *_, Qs = get_flows_from_raw(args.flows_file)

    analyse_autocorrelation(Qs, args.output_file_autocorr)

    plot_history_and_histogram(
        trajectories=trajectories, Qs=Qs,
        **{k: vars(args)[k]
           for k in ('output_file', 'title', 'legend', 'extra_title')}
    )


if __name__ == '__main__':
    main()
