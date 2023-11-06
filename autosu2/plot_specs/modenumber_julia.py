#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.cm import plasma, ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from uncertainties import ufloat

from .common import preliminary

from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, format_value_and_error
from ..do_analysis import get_subdirectory_name
from ..modenumber_julia import read_modenumber_result


def do_plot(data, ensemble=None, filename=None, ax=None):
    gammas = data["raw_gammas"].copy()
    gammas["window_length"] = gammas["xmax"] - gammas["xmin"]

    omega_min = min(gammas.xmin)
    omega_max = max(gammas.xmax)
    if ensemble:
        omega_min = ensemble["measure_modenumber"].get("plot_omega_min", omega_min)
        omega_max = ensemble["measure_modenumber"].get("plot_omega_max", omega_max)

    gammas = gammas.dropna(axis="index", subset=("gamma", "err(stat)"))
    l_min = round(gammas.window_length.min(), 2)
    l_max = round(gammas.window_length.max(), 2)

    # capsize=1 breaks multicolour plots so don't set this here
    if ax:
        ax_supplied = True
    else:
        ax_supplied = False
        set_plot_defaults(linewidth=0.5, capsize=0, preliminary=preliminary)
        fig, ax = plt.subplots(figsize=(3.5, 2))

    colour_norm = LogNorm(vmin=l_min, vmax=l_max)

    colours = plasma(colour_norm(gammas.window_length.values))
    colours[:, 3] *= gammas.weight / max(gammas.weight)

    cbax = inset_axes(ax, width="50%", height="10%", loc="lower right", borderpad=1)
    cb = plt.colorbar(
        ScalarMappable(norm=colour_norm, cmap=plasma),
        cax=cbax,
        orientation="horizontal",
    )
    cbax.text(0.5, 1.75, r"$\Delta\Omega$", ha="center", transform=cbax.transAxes)
    cb.set_ticks((l_min, l_max))
    cb.minorticks_off()
    cb.set_ticklabels((f"{l_min}", f"{l_max}"))
    cbax.xaxis.set_ticks_position("top")
    cbax.xaxis.set_label_position("top")
    cbax.set_in_layout(False)

    if not ax_supplied:
        ax.set_xlabel(r"$\Omega_{\mathrm{LE}}$")
        ax.set_ylabel(r"$\gamma_*$")
    elif ensemble:
        ax.text(
            0.2,
            0.08,
            data["label"],
            horizontalalignment="center",
            verticalalignment="bottom",
            transform=ax.transAxes,
        )

    ax.scatter(gammas.xmin.values, gammas.gamma.values, color=colours, linewidths=0)
    ax.errorbar(
        gammas.xmin.values,
        gammas.gamma.values,
        yerr=gammas["err(stat)"].values,
        linestyle="none",
        marker="None",
        ecolor=colours,
    )

    fit_value = data["gamma"]
    fit_error_statistical = data["gamma_err"]
    fit_error_systematic = data["syst_err"]

    def plot_error_band(centre, error):
        ax.fill_between(
            (
                ensemble["measure_modenumber"]["fit_omega_min"],
                ensemble["measure_modenumber"]["fit_omega_max"],
            ),
            (centre - error, centre - error),
            (centre + error, centre + error),
            color="black",
            alpha=0.2,
            linestyle="None",
            linewidth=0,
        )

    plot_error_band(fit_value, fit_error_statistical)
    plot_error_band(fit_value, fit_error_systematic)

    ax.set_xlim(
        (
            ensemble["measure_modenumber"]["plot_omega_min"],
            ensemble["measure_modenumber"]["plot_omega_max"],
        )
    )
    ax.set_ylim((0, 1.09))

    if not ax_supplied:
        fig.tight_layout(pad=0.08)
        if filename:
            fig.savefig(filename)
            plt.close(fig)
        else:
            plt.show()


def tabulate(fit_results, ensembles):
    filename = "modenumber_gamma.tex"
    columns = (
        "Ensemble",
        None,
        r"$\Omega_{\mathrm{LE}}^{\mathrm{min}}$",
        r"$\Omega_{\mathrm{LE}}^{\mathrm{max}}$",
        r"$\Delta \Omega_{\mathrm{min}}$",
        r"$\Delta \Omega_{\mathrm{max}}$",
        None,
        "$\gamma_*$",
    )
    table_content = []
    table_line = (
        "    {ensemble_name} & {omega_min} & {omega_max} & {len_min} "
        "& {len_max} & {gamma_star}"
    )

    for ensemble_name, gamma_star in fit_results.items():
        ensemble_parameters = ensembles[ensemble_name]["measure_modenumber"]
        table_content.append(
            table_line.format(
                ensemble_name=ensemble_name,
                omega_min=ensemble_parameters["fit_omega_min"],
                omega_max=ensemble_parameters["fit_omega_max"],
                len_min=ensemble_parameters["fit_window_length_min"],
                len_max=ensemble_parameters["fit_window_length_max"],
                gamma_star=format_value_and_error(*gamma_star),
            )
        )

    generate_table_from_content(filename, table_content, columns)


def generate(data, ensembles):
    plot_filename = "final_plots/modenumber.pdf"
    ensembles_to_plot = "DB1M9", "DB4M11", "DB7M10"
    fit_results = {}

    # capsize=1 breaks multicolour plots so don't set this here
    set_plot_defaults(linewidth=0.5, capsize=0, preliminary=preliminary)
    fig, axes = plt.subplots(
        ncols=len(ensembles_to_plot), figsize=(7, 3), squeeze=False
    )

    for ensemble_name, ax in zip(ensembles_to_plot, axes.ravel()):
        modenumber_result = read_modenumber_result(
            f"processed_data/{get_subdirectory_name(ensembles[ensemble_name])}"
            "/modenumber_fit_julia.csv"
        )
        do_plot(modenumber_result, ax=ax, ensemble=ensembles[ensemble_name])
        ax.set_title(
            (
                r"$\beta={beta},am={mass}$"
                + "\n"
                + r"$\Rightarrow\gamma_*={gammas:.02uSL}({systerr})$"
            ).format(
                beta=ensembles[ensemble_name]["beta"],
                mass=ensembles[ensemble_name]["m"],
                gammas=ufloat(
                    modenumber_result["gamma"],
                    modenumber_result["gamma_err"],
                ),
                systerr=int(
                    10 ** (1 - np.floor(np.log10(modenumber_result["gamma_err"])))
                    * modenumber_result["syst_err"]
                ),
            )
        )
    for ax in axes[-1]:
        ax.set_xlabel(r"$\Omega_{\mathrm{LE}}$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\gamma_*$")

    fig.tight_layout(pad=0.08, h_pad=1, w_pad=1)
    fig.savefig(plot_filename)
    plt.close(fig)

    tabulate(fit_results, ensembles)
