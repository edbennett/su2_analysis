from .db import (
    get_measurement,
    measurement_is_up_to_date,
    add_measurement,
)
from .plots import set_plot_defaults

from itertools import product
from re import compile

import lsqfit

import numpy as np
import pandas as pd
import gvar as gv
import itertools

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .modenumber import read_modenumber as read_modenumber_hirep

CONFIGURATION_GETTER = compile(
    r"\[IO\]\[0\]Configuration \[.*n(?P<configuration>[0-9]+)"
)


def read_modenumber(filename, vol, format="hirep"):
    if format == "hirep":
        modenumbers = read_modenumber_hirep(filename)

        omegas = np.asarray(list(modenumbers.keys()))
        nus = [list(nu.values()) for nu in modenumbers.values()]

        nubars = np.transpose(np.array(nus) / vol)

    elif format == "colconf":
        dataraw = np.transpose(np.loadtxt(filename))
        Nmass, Nconf = dataraw.shape

        omegas = dataraw[0, np.arange(0, Nconf, 100)]
        nubars = dataraw[1:, np.arange(0, Nconf, 100)]

        # Normalise for volume of Dirac space
        nubars *= 12

    df = pd.DataFrame(nubars, np.arange(nubars.shape[0]), omegas)

    return df


def set_priors(ensemble_descriptor, boot_gamma):
    mg5 = get_measurement(ensemble_descriptor, "g5_mass")
    mpcac = get_measurement(ensemble_descriptor, "mpcac_mass")

    priors = {
        "log(nu0)": gv.log(gv.gvar("1(1)") * mg5.value),
        "A": gv.gvar(1, 10),
        "gamma": gv.gvar(boot_gamma, 10 * boot_gamma),
        "log(m)": gv.log(gv.gvar("1(1)") * mpcac.value),
    }
    p0 = {
        "nu0": np.exp(priors["log(nu0)"].mean),
        "A": priors["A"].mean,
        "gamma": priors["gamma"].mean,
        "m": 0.0005,
    }
    return priors, p0


def model(x, p):
    exponent = 2 / (1 + p["gamma"])
    x2_minus_m2 = x**2 - p["m"] ** 2
    return p["nu0"] ** 4 + p["A"] * x2_minus_m2**exponent


def window_fit(data, M_min, M_max, priors, p0, Ndata):
    (Omega, Nu) = data
    indices_in_window = [i for i, m in enumerate(Omega) if m <= M_max and m >= M_min]

    if len(indices_in_window) > 5:
        xfit = np.array(Omega[indices_in_window])
        yfit = np.array(Nu[indices_in_window])

        fit = lsqfit.nonlinear_fit(
            data=(xfit, yfit), prior=priors, p0=p0, fcn=model, debug=True
        )

        chi_prior = sum(
            [
                (fit.pmean[k] - priors[k].mean) ** 2 / priors[k].sdev ** 2
                for k in priors.keys()
            ]
        )

        ans = {
            "chi2": fit.chi2 - chi_prior,
            "chipr": chi_prior,
            "pars": fit.p,
            "Ncut": Ndata - len(indices_in_window),
            "Npts": len(xfit),
        }
        return ans
    else:
        return False


def compute_grad(Xmins, Xmaxs, results, delta):
    gmean = np.mean([v["pars"]["gamma"].mean for k, v in results.items()])
    gammas = []
    for xmin, xmax in itertools.product(Xmins, Xmaxs):
        key = (f"{xmin:.3f}", f"{xmax:.3f}")
        val = results[key]["pars"]["gamma"].mean if key in results else gmean
        gammas.append(val)
    gammas = np.array(gammas).reshape((len(Xmins), len(Xmaxs)))
    Grad = np.gradient(gammas, delta)
    Grad2 = np.sqrt(Grad[0] ** 2 + Grad[1] ** 2)

    dgrad = {
        (f"{xmin:.3f}", f"{xmax:.3f}"): Grad2[i, j]
        for (i, xmin), (j, xmax) in itertools.product(
            enumerate(Xmins), enumerate(Xmaxs)
        )
    }

    return dgrad


def windows(filename, format, volume, olow, ohigh, dOM, descriptor, boot_gamma):
    (olow_min, olow_max) = olow
    (ohigh_min, ohigh_max) = ohigh

    # Import data ---------------------------------------------------------------------
    data = read_modenumber(filename, volume, format=format)

    Masses = data.columns.to_numpy()
    Ndata = len([m for m in Masses if m >= olow_min and m <= ohigh_max])
    Nconf = data.shape[0]
    Nus = gv.gvar(data.values.mean(axis=0), np.cov(data.values, rowvar=False) / Nconf)
    Nus = np.array(sorted(Nus))

    # Set priors ----------------------------------------------------------------------
    (priors, p0) = set_priors(descriptor, boot_gamma)

    # Windowing -----------------------------------------------------------------------
    results = {}
    mold = 0
    for M_min in np.arange(olow_min, olow_max, dOM):
        for M_max in np.arange(ohigh_min, ohigh_max, dOM):
            try:
                mok = [m for m in Masses if m <= M_max and m >= M_min]
                if mok != mold:
                    fit = window_fit((Masses, Nus), M_min, M_max, priors, p0, Ndata)
                    if fit:
                        results[(f"{M_min:.3f}", f"{M_max:.3f}")] = fit
                    else:
                        continue
                mold = mok[:]
            except Exception:
                continue

    return results


def AICw(dict, weight="cut"):
    if weight == "cut":
        # chi_aug + 2Ncut
        IC = dict["chi2"] + dict["chipr"] + 2 * dict["Ncut"]
    elif weight == "TIC":
        IC = dict["chi2"] + dict["chipr"] - 2 * (dict["Npts"])
    elif weight == "AIC":
        IC = dict["chi2"] + dict["chipr"]
    return np.exp(-IC / 2)


def weight(results, norm, cutoff, dOM, weight="cut", plot_filter=None):
    # Compute gradient
    xmins = np.unique([float(k[0]) for k in results.keys()])
    xmaxs = np.unique([float(k[1]) for k in results.keys()])
    dGRAD = compute_grad(xmins, xmaxs, results, dOM)

    window_keys = [
        (f"{xmin:.3f}", f"{xmax:.3f}") for xmin, xmax in product(xmins, xmaxs)
    ]
    keyok = [k for k in window_keys if k in results and dGRAD[k] / norm <= cutoff]

    if plot_filter is not None:
        keyok = [k for k in keyok if plot_filter(results[k]["pars"]["gamma"].mean)]

    if not keyok:
        return gv.gvar(0, 0), gv.gvar(0, 0)

    gammastar_samples = np.array([results[k]["pars"]["gamma"] for k in keyok])
    AICW = np.array([AICw(results[k], weight=weight) for k in keyok])
    # normaicw = sum(AICW)
    AICW = AICW / sum(AICW)

    gammast = sum(gammastar_samples * AICW)
    err_syst = np.sqrt(
        sum(gv.mean(gammastar_samples) ** 2 * AICW)
        - sum(gv.mean(gammastar_samples) * AICW) ** 2
    )

    return gammast, gv.gvar(gammast.mean, err_syst)


def grad_plot(results, norm, cutoff, dOM, out=None, plot_filter=None, weight="cut"):
    # Compute gradient
    xmins = np.unique([float(k[0]) for k in results.keys()])
    xmaxs = np.unique([float(k[1]) for k in results.keys()])
    window_keys = [
        (f"{xmin:.3f}", f"{xmax:.3f}")
        for i, xmin in enumerate(xmins)
        for j, xmax in enumerate(xmaxs)
    ]

    dGRAD = compute_grad(xmins, xmaxs, results, dOM)

    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=(7, 4), layout="constrained", subplot_kw={"projection": "3d"}
    )
    engine = fig.get_layout_engine()
    engine.set(rect=(0, 0.0, 1.0, 1.0))

    ax1.set_proj_type("persp", focal_length=0.2)
    ax1.set_box_aspect(None, zoom=0.8)

    map = cm.ScalarMappable(
        cmap=cm.viridis, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=norm)
    )

    # Plot all the point and color them on the base of the gradient
    keys = [k for k in window_keys if k in results]
    xmin = np.array([float(k[0]) for k in keys])
    xmax = np.array([float(k[1]) for k in keys])
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in keys])

    dgamma = np.array([dGRAD[k] for k in keys])

    if plot_filter is None:
        filter_mask = dgamma / norm <= cutoff
    else:
        filter_mask = (dgamma / norm <= cutoff) & plot_filter(gamma)

    ax1.scatter(
        xmax[filter_mask],
        xmin[filter_mask],
        gamma[filter_mask],
        c=map.to_rgba(dgamma[filter_mask]),
        alpha=1,
    )
    ax1.scatter(
        xmax[~filter_mask],
        xmin[~filter_mask],
        gamma[~filter_mask],
        c=map.to_rgba(dgamma[~filter_mask]),
        alpha=0.1,
    )
    bar = ax1.scatter([], [], [], c=map.to_rgba([]))
    ax1ins = inset_axes(
        ax1,
        width="100%",
        height="10%",
        loc="upper left",
        bbox_to_anchor=(0.15, 0.5, 0.7, 0.4),
        bbox_transform=ax1.transAxes,
        borderpad=0,
    )
    fig.colorbar(bar, cax=ax1ins, location="top", shrink=0.6, label="gradient")

    # Plot lines connecting points
    lcolor = "gray"
    lopacity = 0.5
    for xmin in xmins:
        keys = [(f"{xmin:.3f}", f"{k:.3f}") for k in xmaxs]
        keys = [k for k in keys if k in results]

        xmaxs = [float(k[1]) for k in keys]
        xmins = np.full(len(xmaxs), xmin)
        gamma = [results[k]["pars"]["gamma"].mean for k in keys]

        ax1.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)
    for xmax in xmaxs:
        keys = [(f"{k:.3f}", f"{xmax:.3f}") for k in xmins]
        keys = [k for k in keys if k in results]

        xmins = [float(k[0]) for k in keys]
        xmaxs = np.full(len(xmins), xmax)
        gamma = [results[k]["pars"]["gamma"].mean for k in keys]

        ax1.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)

    ax1.tick_params(axis="x", labelrotation=55)
    ax1.tick_params(axis="y", labelrotation=-20)
    ax1.tick_params(axis="z", labelrotation=0)

    ax1.set_xlabel(r"$a\Omega_{\mathrm{max}}$", fontsize=15, labelpad=10)
    ax1.set_ylabel(r"$a\Omega_{\mathrm{min}}$", fontsize=15, labelpad=10)
    ax1.set_zlabel(r"$\gamma^*$", fontsize=15, labelpad=10)

    ax2.set_proj_type("persp", focal_length=0.2)
    ax2.set_box_aspect(None, zoom=0.8)

    keys = [k for k in window_keys if k in results]
    xmin = np.array([float(k[0]) for k in keys])
    xmax = np.array([float(k[1]) for k in keys])
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in keys])
    icwei = np.array([AICw(results[k], weight=weight) for k in keys])

    # Filter points
    kin = [k for k in keys if dGRAD[k] / norm <= cutoff]
    if plot_filter is not None:
        kin = [k for k in kin if plot_filter(results[k]["pars"]["gamma"].mean)]

    cmap = matplotlib.cm.get_cmap("plasma_r")
    if kin:
        aicw = [AICw(results[k], weight=weight) for k in kin]
        mapp = cm.ScalarMappable(
            cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(aicw))
        )

        ax2.scatter(
            xmax[filter_mask],
            xmin[filter_mask],
            gamma[filter_mask],
            c=mapp.to_rgba(icwei[filter_mask]),
            alpha=1,
        )

        ax2.scatter(
            xmax[~filter_mask], xmin[~filter_mask], gamma[~filter_mask], alpha=0.1
        )
        ax2.scatter([], [], [], c=mapp.to_rgba([]))
        ax2ins = inset_axes(
            ax2,
            width="100%",
            height="10%",
            loc="upper left",
            bbox_to_anchor=(0.15, 0.5, 0.7, 0.4),
            bbox_transform=ax2.transAxes,
            borderpad=0,
        )
        fig.colorbar(
            mappable=mapp,
            cax=ax2ins,
            location="top",
            label=r"$e^{-\frac{\mathrm{AIC}}{2}}$",
        )

    # Plot lines connecting points
    lcolor = "gray"
    lopacity = 0.5
    for xmin in xmins:
        keys = [(f"{xmin:.3f}", f"{k:.3f}") for k in xmaxs]
        keys = [k for k in keys if k in results]

        xmaxs = [float(k[1]) for k in keys]
        xmins = np.full(len(xmaxs), xmin)
        gamma = [results[k]["pars"]["gamma"].mean for k in keys]

        ax2.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)
    for xmax in xmaxs:
        keys = [(f"{k:.3f}", f"{xmax:.3f}") for k in xmins]
        keys = [k for k in keys if k in results]

        xmins = [float(k[0]) for k in keys]
        xmaxs = np.full(len(xmins), xmax)
        gamma = [results[k]["pars"]["gamma"].mean for k in keys]

        ax2.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)

    ax2.tick_params(axis="x", labelrotation=55)
    ax2.tick_params(axis="y", labelrotation=-20)
    ax2.tick_params(axis="z", labelrotation=0)

    ax2.set_xlabel(r"$a\Omega_{\mathrm{max}}$", fontsize=15, labelpad=10)
    ax2.set_ylabel(r"$a\Omega_{\mathrm{min}}$", fontsize=15, labelpad=10)
    ax2.set_zlabel(r"$\gamma^*$", fontsize=15, labelpad=10)

    if out is not None:
        fig.savefig(out)


def slice_plot(
    MI,
    MA,
    results,
    gammast,
    err_syst,
    dOM,
    norm,
    cutoff,
    out="./sliceplot.pdf",
    weight="cut",
    plot_filter=None,
):
    # ----------------
    xmins = np.unique([float(k[0]) for k in results.keys()])
    xmaxs = np.unique([float(k[1]) for k in results.keys()])
    window_keys = [
        (f"{xmin:.3f}", f"{xmax:.3f}")
        for i, xmin in enumerate(xmins)
        for j, xmax in enumerate(xmaxs)
    ]

    dGRAD = compute_grad(xmins, xmaxs, results, dOM)

    # Compute overall IC and define color map
    kaux = [k for k in window_keys if k in results if dGRAD[k] / cutoff <= norm]
    aicw = [AICw(results[k], weight="cut") for k in kaux]
    cmap = matplotlib.cm.get_cmap("plasma_r")
    mapp = cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(aicw))
    )

    fig, (ax, ax2) = plt.subplots(
        ncols=2, figsize=(7, 5), layout="constrained", sharey=True
    )

    # FIRST PLOT =====================================================
    omin = xmins[MI]

    # Scatter plot
    kin = [
        k
        for k in window_keys
        if float(k[0]) == omin
        if k in results
        if dGRAD[k] / cutoff <= norm
    ]
    if plot_filter is not None:
        kin = [k for k in kin if plot_filter(results[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[1]) for k in kin]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kin])
    weight = np.array([AICw(results[k], weight="cut") for k in kin])
    ax.scatter(xmaxs, gamma, c=mapp.to_rgba(weight))

    # Scatter out points
    kout = [
        k
        for k in window_keys
        if float(k[0]) == omin
        if k in results
        if not dGRAD[k] / cutoff <= norm
    ]
    if plot_filter is not None:
        kout = [k for k in kout if not plot_filter(results[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[1]) for k in kout]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kout])
    weight = np.array([AICw(results[k], weight="cut") for k in kout])
    ax.scatter(xmaxs, gamma, color="C0", alpha=0.1)

    # Error bands
    kall = [k for k in window_keys if float(k[0]) == omin if k in results]
    xmaxs = [float(k[1]) for k in kall]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kall])
    errg = np.array([results[k]["pars"]["gamma"].sdev for k in kall])
    ax.fill_between(xmaxs, gamma - errg, gamma + errg, color="gray", alpha=0.1)

    # Result span
    ax.axhspan(
        gammast.mean + gammast.sdev,
        gammast.mean - gammast.sdev,
        alpha=0.2,
        label=r"$\gamma^*$".format(),
    )
    ax.axhspan(gammast.mean + err_syst, gammast.mean - err_syst, alpha=0.2)

    ax.set_xlabel(r"$a\Omega_{\mathrm{max}}$")
    ax.set_ylabel(r"$\gamma^*$")
    ax.set_xlim(xmin=min(xmaxs), xmax=max(xmaxs))
    if not np.isnan(gammast.mean):
        ax.set_ylim(ymin=gammast.mean - 0.25, ymax=gammast.mean + 0.25)

    ax1 = plt.twinx(ax)
    chi2 = np.array([results[k]["chi2"] / (results[k]["Npts"]) for k in kall])
    ax1.plot(xmaxs, chi2, color="C1", label=r"$\frac{\chi^2}{\mathrm{dof}}$", alpha=0.2)

    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax1.get_legend_handles_labels()
    ax.legend(lines_left + lines_right, labels_left + labels_right, loc="best")

    ax1.set_ylabel(r"$\frac{\chi^2}{\mathrm{dof}}$")
    ax1.set_title(r"$a\Omega_{{\mathrm{{min}}}} = {{{0}}}$".format(f"{omin:.3f}"))

    # SECOND PLOT =====================================================
    omax = xmaxs[MA]

    # Scatter plot
    kin = [
        k
        for k in window_keys
        if float(k[1]) == omax
        if k in results
        if dGRAD[k] / cutoff <= norm
    ]
    if plot_filter is not None:
        kin = [k for k in kin if plot_filter(results[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[0]) for k in kin]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kin])
    weight = np.array([AICw(results[k], weight="cut") for k in kin])
    ax2.scatter(xmaxs, gamma, c=mapp.to_rgba(weight))

    # Scatter out points
    kout = [
        k
        for k in window_keys
        if float(k[1]) == omax
        if k in results
        if not dGRAD[k] / cutoff <= norm
    ]
    if plot_filter is not None:
        kout = [k for k in kout if not plot_filter(results[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[0]) for k in kout]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kout])
    weight = np.array([AICw(results[k], weight="cut") for k in kout])
    ax2.scatter(xmaxs, gamma, color="C0", alpha=0.1)

    # Error bands
    kall = [k for k in window_keys if float(k[1]) == omax if k in results]
    xmaxs = [float(k[0]) for k in kall]
    gamma = np.array([results[k]["pars"]["gamma"].mean for k in kall])
    errg = np.array([results[k]["pars"]["gamma"].sdev for k in kall])
    ax2.fill_between(xmaxs, gamma - errg, gamma + errg, color="gray", alpha=0.1)

    # Result span
    ax2.axhspan(
        gammast.mean + gammast.sdev,
        gammast.mean - gammast.sdev,
        alpha=0.2,
        label=r"$\gamma^*$".format(),
    )
    ax2.axhspan(gammast.mean + err_syst, gammast.mean - err_syst, alpha=0.2)

    ax2.set_xlabel(r"$a\Omega_{\mathrm{max}}$")
    ax2.set_ylabel(r"$\gamma^*$")
    ax2.set_xlim(xmin=min(xmaxs), xmax=max(xmaxs))
    if not np.isnan(gammast.mean):
        ax2.set_ylim(ymin=gammast.mean - 0.25, ymax=gammast.mean + 0.25)

    ax3 = plt.twinx(ax2)
    ax3.sharey(ax1)
    chi2 = np.array([results[k]["chi2"] / (results[k]["Npts"]) for k in kall])
    ax3.plot(xmaxs, chi2, color="C1", label=r"$\frac{\chi^2}{\mathrm{dof}}$", alpha=0.2)
    ax3.set_ylim(ymin=0, ymax=2)

    lines2_left, labels2_left = ax2.get_legend_handles_labels()
    lines2_right, labels2_right = ax3.get_legend_handles_labels()
    ax2.legend(lines2_left + lines2_right, labels2_left + labels2_right, loc="best")

    ax3.set_ylabel(r"$\frac{\chi^2}{\mathrm{dof}}$")
    ax3.set_title(r"$a\Omega_{{\mathrm{{min}}}} = {{{0}}}$".format(f"{omax:.3f}"))

    fig.savefig(out)


def do_modenumber_fit_aic(ensemble, filename, boot_gamma, plot_directory):
    set_plot_defaults()
    if (
        ensemble
        and measurement_is_up_to_date(
            ensemble["descriptor"], "gamma_aic", compare_file=filename
        )
        and measurement_is_up_to_date(
            ensemble["descriptor"], "gamma_aic_syst", compare_file=filename
        )
    ):
        # Already up to date
        return

    V = ensemble["T"] * ensemble["L"] ** 3
    pars = ensemble["measure_modenumber"]

    olow = (pars["fit_omega_min"], pars["fit_omega_max"])
    do = (pars["fit_window_length_min"], pars["fit_window_length_max"])
    ohigh = (min(olow) + min(do), max(olow) + max(do))
    delta = pars["delta_fit"]

    results = windows(
        filename,
        pars["format"],
        V,
        olow,
        ohigh,
        delta,
        ensemble["descriptor"],
        boot_gamma,
    )

    def f(m):
        return lambda g: g < m

    gstat, gsyst = weight(
        results,
        pars["norm"],
        pars["cutoff"],
        delta,
        weight="cut",
        plot_filter=f(pars["filter"]),
    )

    grad_plot(
        results,
        pars["norm"],
        pars["cutoff"],
        delta,
        plot_filter=f(pars["filter"]),
        out=f"{plot_directory}/modenumber_grad_plot.pdf",
    )
    slice_plot(
        pars["plot_omega_min"],
        pars["plot_omega_max"],
        results,
        gsyst,
        gstat.sdev,
        delta,
        pars["norm"],
        pars["cutoff"],
        weight="cut",
        plot_filter=f(pars["filter"]),
        out=f"{plot_directory}/modenumber_slice_plot.pdf",
    )

    if ensemble:
        add_measurement(ensemble["descriptor"], "gamma_aic", gstat.mean, gstat.sdev)
        add_measurement(ensemble["descriptor"], "gamma_aic_syst", gsyst.sdev)

    return gstat, gsyst
