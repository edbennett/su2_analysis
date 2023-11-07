import matplotlib.pyplot as plt
import numpy as np
import argparse

from flow_analysis.fit_forms import gaussian, exp_decay
from flow_analysis.readers import readers
from flow_analysis.measurements.Q import Q_mean, Q_fit, Q_susceptibility, flat_bin_Qs
from flow_analysis.stats.autocorrelation import autocorr, exp_autocorrelation_fit

from .data import get_flows_from_raw
from .data import file_is_up_to_date
from .db import measurement_is_up_to_date, add_measurement
from .plots import set_plot_defaults


def analyse_autocorrelation(series, filename, fit_range=10):
    """
    Calculates and plots the autocorrelation function of `series`.
    Outputs to `filename`, or to screen if this is not specified.
    Fits the exponential autocorrelation time of the first `fit_range` points
    of `series` and returns it with its uncertainty.
    """

    set_plot_defaults()
    f, ax = plt.subplots()

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$C(\tau)$")

    acf = autocorr(series)
    ax.scatter(np.arange(len(acf)), acf)

    tau_exp = exp_autocorrelation_fit(series, fit_range=fit_range)
    acf_domain = np.linspace(0, len(acf), 1000)
    ax.plot(acf_domain, exp_decay(acf_domain, tau_exp.nominal_value))

    if filename:
        f.savefig(filename)
    else:
        plt.show()
    plt.close(f)

    return tau_exp


def plot_history_and_histogram(
    flows,
    output_file=None,
    title=True,
    extra_title="",
    legend=False,
    history_ax=None,
    histogram_ax=None,
    label_axes=True,
    count_axis="absolute",
):
    """Histograms, amd plots to the screen or a PDF if specified,
    the topological charges in Qs, with trajectory numbers provided in
    `trajectories`."""

    assert (not history_ax and not histogram_ax) or (history_ax and histogram_ax)
    assert not (output_file and history_ax)

    f = None
    if not history_ax:
        set_plot_defaults()
        f, (history_ax, histogram_ax) = plt.subplots(
            1, 2, sharey=True, gridspec_kw={"width_ratios": [3, 1]}, figsize=(5.5, 2.5)
        )
        if legend:
            history_ax.legend(
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
                loc=3,
                ncol=5,
                mode="expand",
                borderaxespad=0.0,
                frameon=False,
            )

    if label_axes:
        history_ax.set_xlabel("Trajectory")
        histogram_ax.set_xlabel("Count")

    history_ax.set_ylabel("$Q$")

    Qs = flows.Q_history()
    for ensemble_trajectories, Q_ensemble in zip(
        flows.group(flows.trajectories),
        flows.group(Qs),
    ):
        history_ax.step(ensemble_trajectories, Q_ensemble)

    Q_range, Q_counts = flat_bin_Qs(Qs)
    A, Q0, sigma = Q_fit(flows, with_amplitude=True)

    history_ax.set_ylim(min(Q_range) - 0.5, max(Q_range) + 0.5)

    if count_axis == "relative":
        total_count = sum(Q_counts)
        Q_counts = Q_counts / total_count
        A /= total_count

    histogram_ax.step(Q_counts, Q_range - 0.5, label="Histogram")

    smooth_Q_range = np.linspace(min(Q_range) - 0.5, max(Q_range) + 0.5, 1000)
    histogram_ax.plot(
        gaussian(
            smooth_Q_range, A.nominal_value, Q0.nominal_value, sigma.nominal_value
        ),
        smooth_Q_range,
        label="Fit",
    )

    if f:
        f.tight_layout()

        if title:
            f.suptitle(
                ("" if extra_title is None else extra_title)
                + r" $Q_0 = {:.2uSL}$; $\sigma = {:.2uSL}$".format(Q0, abs(sigma))
            )
            f.subplots_adjust(top=0.8 if legend else 0.9)

        if output_file is None:
            plt.show()
        else:
            f.savefig(output_file)
        plt.close(f)

    return Q0, abs(sigma)


def plot_measure_and_save_Q(
    flows_file,
    simulation_descriptor=None,
    output_file_history=None,
    output_file_autocorr=None,
    reader="hirep",
):
    """Reads in flows_file in HiRep~MILC format, calculates average
    Q and topological susceptibility, plots and histograms, saves
    to the database."""

    if file_is_up_to_date(output_file_history, compare_file=flows_file):
        if simulation_descriptor:
            if measurement_is_up_to_date(
                simulation_descriptor, "fitted_Q0", compare_file=flows_file
            ) and measurement_is_up_to_date(
                simulation_descriptor, "Q_width", compare_file=flows_file
            ):
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
            simulation_descriptor, "Q0", compare_file=flows_file
        ) and measurement_is_up_to_date(
            simulation_descriptor, "chi_top", compare_file=flows_file
        ):
            do_bootstrap = False
        else:
            do_bootstrap = True
    else:
        do_bootstrap = True

    result = {}

    if do_fit_and_plot or do_bootstrap:
        flows = readers[reader](flows_file)
        tau_exp = analyse_autocorrelation(flows.Q_history(), output_file_autocorr)

        fit_range = 20
        while tau_exp.nominal_value > fit_range and fit_range < len(flows):
            tau_exp = analyse_autocorrelation(
                flows, output_file_autocorr, fit_range=fit_range
            )
            fit_range *= 2

        if simulation_descriptor:
            add_measurement(simulation_descriptor, "Q_tau_exp", tau_exp)

    if do_fit_and_plot:
        fitted_Q0, Q_width = plot_history_and_histogram(
            flows, output_file=output_file_history, title=False
        )
        if simulation_descriptor:
            add_measurement(simulation_descriptor, "fitted_Q0", fitted_Q0)
            add_measurement(simulation_descriptor, "Q_width", Q_width)
        result["fitted_Q0"] = fitted_Q0
        result["Q_width"] = Q_width

    if do_bootstrap:
        Q0 = Q_mean(flows)
        add_measurement(simulation_descriptor, "Q0", Q0)
        result["Q0"] = Q0

        chi_top = Q_susceptibility(flows)
        add_measurement(simulation_descriptor, "chi_top", chi_top)
        result["chi_top"] = chi_top

    return result


def main():
    """Runs the program"""

    parser = argparse.ArgumentParser(
        description="Plot the topological " "charge history and histogram"
    )
    parser.add_argument("flows_file", help="A Q history in HiRep~MILC format")
    parser.add_argument(
        "--output_file",
        default=None,
        help="PDF file to output history and histogram to. "
        "If omitted, outputs to the screen",
    )
    parser.add_argument(
        "--output_file_autocorr",
        default=None,
        help="PDF file to output autocorrelation plot to. "
        "If omitted, outputs to the screen",
    )
    parser.add_argument(
        "--no_title", action="store_false", dest="title", help="Disable the plot title."
    )
    parser.add_argument(
        "--with_legend",
        action="store_true",
        dest="legend",
        help="Enable the legend on the history plot.",
    )
    parser.add_argument("--extra_title", help="Extra text to add to the title")
    args = parser.parse_args()

    trajectories, *_, Qs = get_flows_from_raw(args.flows_file)

    analyse_autocorrelation(Qs, args.output_file_autocorr)

    print(
        plot_history_and_histogram(
            trajectories=trajectories,
            Qs=Qs,
            **{
                k: vars(args)[k]
                for k in ("output_file", "title", "legend", "extra_title")
            },
        )
    )


if __name__ == "__main__":
    main()
