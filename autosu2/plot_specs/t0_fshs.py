#!/usr/bin/env python3

import logging

from flow_analysis.readers import readers
from flow_analysis.measurements.scales import (
    bootstrap_finalize,
    compute_wt_samples,
    threshold_interpolate,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from uncertainties import ufloat

from ..data import get_subdirectory_name
from ..db import describe_ensemble, get_measurement
from ..plots import set_plot_defaults

from .common import preliminary
from .fshs import sm_residual


class NothingToSeeError(Exception):
    pass


def lbgfsb_uncertainty(result):
    lbgfsb_ftol = 2.22e-9
    return (result.hess_inv([1])[0] * lbgfsb_ftol * max(1, abs(result.fun))) ** 0.5


def zip_dicts(*dicts):
    # As suggested by
    # https://stackoverflow.com/questions/16458340/python-equivalent-of-zip-for-dictionaries

    if not dicts:
        return
    for key in set(dicts[0]).intersection(*dicts[1:]):
        yield (key,) + tuple(d[key] for d in dicts)


def amap(function, iterable):
    return np.asarray(list(map(function, iterable)))


def fit_fshs(ensembles, mpcacs, w0s, ax=None, label=None):
    data = pd.DataFrame(
        [
            {"L": ensemble["L"], "value_mpcac_mass": mpcac, "one_over_w0": 1 / w0}
            for label, ensemble, mpcac, w0 in zip_dicts(ensembles, mpcacs, w0s)
        ]
    )
    try:
        result = minimize(
            lambda gamma: sm_residual(gamma, data, observable="one_over_w0"),
            1.0,
            method="L-BFGS-B",
        )
    except IndexError:
        return None

    if ax:
        ax.scatter(
            data.L * data.value_mpcac_mass ** (1 / (1 + result.x[0])),
            data.L * data.one_over_w0,
            label=label,
        )
    return result


def plot_scan(results_df, metafit_params, filename):
    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlabel(r"1 / $\mathcal{E}_0$")
    ax.set_ylabel(r"$\gamma_*$")

    ax.errorbar(
        results_df.recip_scale,
        results_df.gamma,
        yerr=results_df.gamma_err,
        label="FSHS fit results",
        ls="none",
        marker="x",
    )
    metafit_range = np.linspace(0, max(results_df.recip_scale) * 1.1, 1000)
    ax.plot(
        metafit_range,
        exp_fit_form(metafit_range, *metafit_params),
        label="Fit of fits",
    )
    ax.set_ylim(-0.1 * max(results_df.gamma), 1.1 * max(results_df.gamma))
    ax.legend(loc="best")
    fig.savefig(filename)
    plt.close(fig)


def exp_fit_form(x, A, B, C):
    return A * np.exp(B * x) + C


def metafit(results_df):
    try:
        popt, pcov = curve_fit(
            exp_fit_form,
            results_df.recip_scale,
            results_df.gamma,
            sigma=results_df.gamma_weighted_err,
            absolute_sigma=False,
        )
    except RuntimeError:
        return np.nan, [np.nan, np.nan, np.nan], np.nan
    else:
        gamma_zero_value = exp_fit_form(0, *popt)
        gamma_zero_err = (pcov[0, 0] + pcov[2, 2]) ** 0.5
        gamma_zero = ufloat(gamma_zero_value, gamma_zero_err)
        chi2 = sum(
            (exp_fit_form(results_df.recip_scale, *popt) - results_df.gamma) ** 2
            / results_df.gamma_err**2
        ) / (len(results_df) - 2)
        return gamma_zero, popt, chi2


def results_to_df(fit_results):
    return pd.DataFrame(
        [
            {
                "scale": scale,
                "recip_scale": 1 / scale,
                "gamma": result.x[0],
                "gamma_err": lbgfsb_uncertainty(result),
                "residual": result.fun,
                "gamma_weighted_err": lbgfsb_uncertainty(result) * result.fun,
            }
            for scale, result in fit_results.items()
            if result
        ]
    )


def w0s_at_scale(flow_ensembles, bs_samples, scale):
    result = {}
    for label, sample, flow_ensemble in zip_dicts(bs_samples, flow_ensembles):
        try:
            result[label] = bootstrap_finalize(
                threshold_interpolate(
                    flow_ensemble,
                    sample,
                    scale,
                )
            ).nominal_value
        except ValueError:
            message = f"No points reach threshold for {label}, {scale}"
            logging.warning(message)
    return result


def generate_single(ensembles, sampleplot_filename, scanplot_filename, scales):
    logging.info("Getting PCAC masses")
    mpcacs = {}
    for label in ensembles:
        try:
            ensemble = ensembles[label]
            mpcacs[label] = get_measurement(
                describe_ensemble(ensemble, label), "mpcac_mass"
            ).value
        except KeyError:
            message = f"No PCAC mass found for {label}"
            logging.warn(message)
            continue

    logging.info("Loading flows")
    flow_ensembles = {
        label: readers[ensemble["measure_gflow"]](
            f"raw_data/{get_subdirectory_name(ensemble)}/out_wflow"
        )
        for label, ensemble in ensembles.items()
    }

    # Sets of bootstrap samples of w(t) for each ensemble
    logging.info("Computing bootstrap samples")
    wt_bs_samples = {
        label: compute_wt_samples(flow_ensemble, operator="sym")
        for label, flow_ensemble in flow_ensembles.items()
    }

    fig, ax = plt.subplots(layout="constrained")
    ax.set_xlabel(r"$Lm_{\mathrm{PCAC}}^{1/(1+\gamma_*)}$")
    ax.set_ylabel(r"$L/w_0$")

    fit_results = {
        scale: fit_fshs(
            ensembles,
            mpcacs,
            w0s_at_scale(flow_ensembles, wt_bs_samples, scale),
            ax=ax,
            label=f"{scale:.02f}",
        )
        for scale in scales
    }
    ax.legend(loc="best")
    fig.savefig(sampleplot_filename)

    results_df = results_to_df(fit_results)
    if len(results_df) == 0:
        raise NothingToSeeError
    gamma, metafit_params, metafit_chi2 = metafit(results_df)
    plot_scan(results_df, metafit_params, scanplot_filename)


def filter_ensembles(ensembles, **kwargs):
    result = {
        label: ensemble
        for label, ensemble in ensembles.items()
        if ensemble.get("measure_gflow")
    }
    for key, value in kwargs.items():
        result = {
            label: ensemble
            for label, ensemble in result.items()
            if ensemble.get(key) == value
        }
    return result


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.3, preliminary=preliminary)

    for beta in []:  # 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4:
        logging.info(f"Nf=1, beta={beta}...")
        generate_single(
            filter_ensembles(ensembles, Nf=1, beta=beta),
            sampleplot_filename=f"auxiliary_plots/w0_fshs_sample_Nf1_beta{beta}.pdf",
            scanplot_filename=f"auxiliary_plots/w0_fshs_Nf1_beta{beta}.pdf",
            scales=np.arange(0.15, 0.5, 0.01),
        )

    logging.info("Nf=2, beta=2.35...")
    try:
        generate_single(
            filter_ensembles(ensembles, Nf=2, beta=2.35),
            sampleplot_filename="auxiliary_plots/w0_fshs_sample_Nf2_beta2.35.pdf",
            scanplot_filename="auxiliary_plots/w0_fshs_Nf2_beta2.35.pdf",
            scales=np.arange(0.1, 0.2, 0.01),
        )
    except NothingToSeeError:
        logging.warn("Nothing plottable found for Nf=2, beta=2.35")
