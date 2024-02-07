from .db import (
    get_measurement,
    measurement_is_up_to_date,
    add_measurement,
)


from re import compile

import lsqfit

import numpy as np
import pandas as pd
import gvar as gv
import itertools

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm

from .modenumber import read_modenumber as read_modenumber_hirep

CONFIGURATION_GETTER = compile(
    r"\[IO\]\[0\]Configuration \[.*n(?P<configuration>[0-9]+)"
)


def read_modenumber(filename, vol, format="hirep"):
    if format == "hirep":
        modenumbers = read_modenumber_hirep(filename)

        OMEGA, y = [], []
        for m, nu in modenumbers.items():
            OMEGA.append(m)
            y.append(list(nu.values()))
        data = np.transpose(np.array(y) / vol)

    elif format == "colconf":
        dataraw = np.transpose(np.loadtxt(filename))
        Nmass, Nconf = dataraw.shape

        OMEGA = dataraw[0, np.arange(0, Nconf, 100)]
        data = dataraw[1:, np.arange(0, Nconf, 100)]

    df = pd.DataFrame(data, np.arange(data.shape[0]), OMEGA)

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


def Model(x, p):
    exponent = 2 / (1 + p["gamma"])
    x2_minus_m2 = x**2 - p["m"] ** 2
    return p["nu0"] ** 4 + p["A"] * x2_minus_m2**exponent


def window_fit(data, M_min, M_max, priors, p0, Ndata):
    (Omega, Nu) = data
    iiok = [i for i, m in enumerate(Omega) if m <= M_max and m >= M_min]

    if len(iiok) > 5:
        xfit = np.array(Omega[iiok])
        yfit = np.array(Nu[iiok])

        fit = lsqfit.nonlinear_fit(
            data=(xfit, yfit), prior=priors, p0=p0, fcn=Model, debug=True
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
            "Ncut": Ndata - len(iiok),
            "Npts": len(xfit),
        }
        return ans
    else:
        return False


def compute_grad(Xmins, Xmaxs, RESULTS, delta):
    gmean = np.mean([v["pars"]["gamma"].mean for k, v in RESULTS.items()])
    gammas = []
    for xmin, xmax in itertools.product(Xmins, Xmaxs):
        key = (f"{xmin:.3f}", f"{xmax:.3f}")
        val = RESULTS[key]["pars"]["gamma"].mean if key in RESULTS else gmean
        gammas.append(val)
    gammas = np.array(gammas).reshape((len(Xmins), len(Xmaxs)))
    Grad = np.gradient(gammas, delta)
    Grad2 = np.sqrt(Grad[0] ** 2 + Grad[1] ** 2)

    # dGRAD = {}
    # for i, xmin in enumerate(Xmins):
    #     for j, xmax in enumerate(Xmaxs):
    #         key = (f"{xmin:.3f}", f"{xmax:.3f}")
    #         dGRAD[key] = Grad2[i, j]

    dgrad = {
        (f"{xmin: .3f}", f"{xmax:.3f}"): Grad2[i, j]
        for (i, xmin), (j, xmax) in itertools.product(
            enumerate(Xmins), enumerate(Xmaxs)
        )
    }

    return dgrad


def WINDOWS(filename, format, volume, OLOW, OHIGH, dOM, descriptor, boot_gamma):
    (OLOW_MIN, OLOW_MAX) = OLOW
    (OHIGH_MIN, OHIGH_MAX) = OHIGH

    # Import data ---------------------------------------------------------------------
    data = read_modenumber(filename, volume, format=format)

    Masses = data.columns.to_numpy()
    Ndata = len([m for m in Masses if m >= OLOW_MIN and m <= OHIGH_MAX])
    Nconf = data.shape[0]
    Nus = gv.gvar(data.values.mean(axis=0), np.cov(data.values, rowvar=False) / Nconf)
    Nus = np.array(sorted(Nus))

    # Set priors ----------------------------------------------------------------------
    (priors, p0) = set_priors(descriptor, boot_gamma)
    print(priors, p0)

    # Windowing -----------------------------------------------------------------------
    RESULTS = {}
    mold = 0
    for M_min in np.arange(OLOW_MIN, OLOW_MAX, dOM):
        for M_max in np.arange(OHIGH_MIN, OHIGH_MAX, dOM):
            try:
                mok = [m for m in Masses if m <= M_max and m >= M_min]
                if mok != mold:
                    fit = window_fit((Masses, Nus), M_min, M_max, priors, p0, Ndata)
                    if not fit:
                        continue
                    else:
                        RESULTS[(f"{M_min:.3f}", f"{M_max:.3f}")] = fit
                mold = mok[:]
            except Exception:
                continue

    return RESULTS


def AICw(dict, weight="cut"):
    if weight == "cut":
        # chi_aug + 2Ncut
        IC = dict["chi2"] + dict["chipr"] + 2 * dict["Ncut"]
    elif weight == "TIC":
        IC = dict["chi2"] + dict["chipr"] - 2 * (dict["Npts"])
    elif weight == "AIC":
        IC = dict["chi2"] + dict["chipr"]
    return np.exp(-IC / 2)


def WEIGHT(RESULTS, NORM, CUTOFF, dOM, weight="cut", FILTER=None):
    # Compute gradient
    XMINS = np.unique([float(k[0]) for k in RESULTS.keys()])
    XMAXS = np.unique([float(k[1]) for k in RESULTS.keys()])
    dGRAD = compute_grad(XMINS, XMAXS, RESULTS, dOM)

    KEYS = [
        (f"{xmin:.3f}", f"{xmax:.3f}")
        for i, xmin in enumerate(XMINS)
        for j, xmax in enumerate(XMAXS)
    ]
    keyok = [k for k in KEYS if k in RESULTS and dGRAD[k] / NORM <= CUTOFF]

    if FILTER is not None:
        keyok = [k for k in keyok if FILTER(RESULTS[k]["pars"]["gamma"].mean)]

    g = np.array([RESULTS[k]["pars"]["gamma"] for k in keyok])
    AICW = np.array([AICw(RESULTS[k], weight=weight) for k in keyok])
    # NORMaicw = sum(AICW)
    AICW = AICW / sum(AICW)

    gammast = sum(g * AICW)
    err_syst = np.sqrt(sum(gv.mean(g) ** 2 * AICW) - sum(gv.mean(g) * AICW) ** 2)

    return gammast, gv.gvar(gammast.mean, err_syst)


def grad_plot(RESULTS, NORM, CUTOFF, dOM, out=None, FILTER=None, weight="cut"):
    # Compute gradient
    XMINS = np.unique([float(k[0]) for k in RESULTS.keys()])
    XMAXS = np.unique([float(k[1]) for k in RESULTS.keys()])
    KEYS = [
        (f"{xmin:.3f}", f"{xmax:.3f}")
        for i, xmin in enumerate(XMINS)
        for j, xmax in enumerate(XMAXS)
    ]

    dGRAD = compute_grad(XMINS, XMAXS, RESULTS, dOM)

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    fig = plt.figure(figsize=(10, 10))

    ax1 = fig.add_subplot(2, 1, 1, projection="3d")
    ax1.set_proj_type("persp", focal_length=0.2)

    map = cm.ScalarMappable(
        cmap=cm.viridis, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=NORM)
    )

    # Plot all the point and color them on the base of the gradient
    keys = [k for k in KEYS if k in RESULTS]
    xmin = np.array([float(k[0]) for k in keys])
    xmax = np.array([float(k[1]) for k in keys])
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in keys])

    dgamma = np.array([dGRAD[k] for k in keys])
    # keyok = [k for k in keys if dGRAD[k] / NORM <= CUTOFF]

    if FILTER is None:
        iyes = [k / NORM <= CUTOFF for k in dgamma]
    else:
        iyes = [d / NORM <= CUTOFF and FILTER(g) for (d, g) in zip(dgamma, gamma)]
    ino = [not i for i in iyes]

    ax1.scatter(
        xmax[iyes], xmin[iyes], gamma[iyes], c=map.to_rgba(dgamma[iyes]), alpha=1
    )
    ax1.scatter(xmax[ino], xmin[ino], gamma[ino], c=map.to_rgba(dgamma[ino]), alpha=0.1)
    p = ax1.scatter([], [], [], c=map.to_rgba([]))
    fig.colorbar(p, ax=ax1, shrink=0.6, label="gradient")

    # Plot lines connecting points
    lcolor = "gray"
    lopacity = 0.5
    for xmin in XMINS:
        keys = [(f"{xmin:.3f}", f"{k:.3f}") for k in XMAXS]
        keys = [k for k in keys if k in RESULTS]

        xmaxs = [float(k[1]) for k in keys]
        xmins = np.full(len(xmaxs), xmin)
        gamma = [RESULTS[k]["pars"]["gamma"].mean for k in keys]

        ax1.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)
    for xmax in XMAXS:
        keys = [(f"{k:.3f}", f"{xmax:.3f}") for k in XMINS]
        keys = [k for k in keys if k in RESULTS]

        xmins = [float(k[0]) for k in keys]
        xmaxs = np.full(len(xmins), xmax)
        gamma = [RESULTS[k]["pars"]["gamma"].mean for k in keys]

        ax1.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)

    # ax1.set_xticks(rotation=90)
    ax1.tick_params(axis="x", labelrotation=55)
    ax1.tick_params(axis="y", labelrotation=-20)
    ax1.tick_params(axis="z", labelrotation=0)

    ax1.set_xlabel(r"$a\Omega_{max}$", fontsize=15)
    ax1.set_ylabel(r"$a\Omega_{min}$", fontsize=15)
    ax1.set_zlabel(r"$\gamma^*$", fontsize=15)

    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2, projection="3d")
    ax2.set_proj_type("persp", focal_length=0.2)

    keys = [k for k in KEYS if k in RESULTS]
    xmin = np.array([float(k[0]) for k in keys])
    xmax = np.array([float(k[1]) for k in keys])
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in keys])
    icwei = np.array([AICw(RESULTS[k], weight=weight) for k in keys])

    # Filter points
    kin = [k for k in keys if dGRAD[k] / NORM <= CUTOFF]
    if FILTER is not None:
        kin = [k for k in kin if FILTER(RESULTS[k]["pars"]["gamma"].mean)]
    aicw = [AICw(RESULTS[k], weight=weight) for k in kin]
    cmap = matplotlib.cm.get_cmap("plasma_r")
    mapp = cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(aicw))
    )

    ax2.scatter(
        xmax[iyes], xmin[iyes], gamma[iyes], c=mapp.to_rgba(icwei[iyes]), alpha=1
    )

    ax2.scatter(xmax[ino], xmin[ino], gamma[ino], alpha=0.1)
    ax2.scatter([], [], [], c=mapp.to_rgba([]))
    fig.colorbar(mappable=mapp, ax=ax2, shrink=0.6, label=r"$e^{-\frac{AIC}{2}}$")

    # Plot lines connecting points
    lcolor = "gray"
    lopacity = 0.5
    for xmin in XMINS:
        keys = [(f"{xmin:.3f}", f"{k:.3f}") for k in XMAXS]
        keys = [k for k in keys if k in RESULTS]

        xmaxs = [float(k[1]) for k in keys]
        xmins = np.full(len(xmaxs), xmin)
        gamma = [RESULTS[k]["pars"]["gamma"].mean for k in keys]

        ax2.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)
    for xmax in XMAXS:
        keys = [(f"{k:.3f}", f"{xmax:.3f}") for k in XMINS]
        keys = [k for k in keys if k in RESULTS]

        xmins = [float(k[0]) for k in keys]
        xmaxs = np.full(len(xmins), xmax)
        gamma = [RESULTS[k]["pars"]["gamma"].mean for k in keys]

        ax2.plot(xmaxs, xmins, gamma, color=lcolor, alpha=lopacity)

    ax2.tick_params(axis="x", labelrotation=55)
    ax2.tick_params(axis="y", labelrotation=-20)
    ax2.tick_params(axis="z", labelrotation=0)

    ax2.set_xlabel(r"$a\Omega_{max}$", fontsize=15)
    ax2.set_ylabel(r"$a\Omega_{min}$", fontsize=15)
    ax2.set_zlabel(r"$\gamma^*$", fontsize=15)
    ax2.legend()

    # fig.tight_layout()
    # plt.tight_layout()

    if out is not None:
        fig.savefig(out)


def slice_plot(
    MI,
    MA,
    RESULTS,
    gammast,
    err_syst,
    dOM,
    NORM,
    CUTOFF,
    out="./sliceplot.pdf",
    weight="cut",
    FILTER=None,
):
    # ----------------
    XMINS = np.unique([float(k[0]) for k in RESULTS.keys()])
    XMAXS = np.unique([float(k[1]) for k in RESULTS.keys()])
    KEYS = [
        (f"{xmin:.3f}", f"{xmax:.3f}")
        for i, xmin in enumerate(XMINS)
        for j, xmax in enumerate(XMAXS)
    ]

    dGRAD = compute_grad(XMINS, XMAXS, RESULTS, dOM)

    # Compute overall IC and define color map
    kaux = [k for k in KEYS if k in RESULTS if dGRAD[k] / CUTOFF <= NORM]
    aicw = [AICw(RESULTS[k], weight="cut") for k in kaux]
    cmap = matplotlib.cm.get_cmap("plasma_r")
    mapp = cm.ScalarMappable(
        cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0.0, vmax=max(aicw))
    )

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.size"] = 12

    fig = plt.figure(figsize=(5, 8))

    # FIRST PLOT =====================================================
    ax = fig.add_subplot(2, 1, 1)

    OMIN = XMINS[MI]

    # Scatter plot
    kin = [
        k
        for k in KEYS
        if float(k[0]) == OMIN
        if k in RESULTS
        if dGRAD[k] / CUTOFF <= NORM
    ]
    if FILTER is not None:
        kin = [k for k in kin if FILTER(RESULTS[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[1]) for k in kin]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kin])
    weight = np.array([AICw(RESULTS[k], weight="cut") for k in kin])
    ax.scatter(xmaxs, gamma, c=mapp.to_rgba(weight))

    # Scatter out points
    kout = [
        k
        for k in KEYS
        if float(k[0]) == OMIN
        if k in RESULTS
        if not dGRAD[k] / CUTOFF <= NORM
    ]
    if FILTER is not None:
        kout = [k for k in kout if not FILTER(RESULTS[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[1]) for k in kout]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kout])
    weight = np.array([AICw(RESULTS[k], weight="cut") for k in kout])
    ax.scatter(xmaxs, gamma, color="C0", alpha=0.1)

    # Error bands
    kall = [k for k in KEYS if float(k[0]) == OMIN if k in RESULTS]
    xmaxs = [float(k[1]) for k in kall]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kall])
    errg = np.array([RESULTS[k]["pars"]["gamma"].sdev for k in kall])
    ax.fill_between(xmaxs, gamma - errg, gamma + errg, color="gray", alpha=0.1)

    # Result span
    ax.axhspan(
        gammast.mean + gammast.sdev,
        gammast.mean - gammast.sdev,
        alpha=0.2,
        label=r"$\gamma^*$".format(),
    )
    ax.axhspan(gammast.mean + err_syst, gammast.mean - err_syst, alpha=0.2)

    ax.set_xlabel(r"$a\Omega_{max}$")
    ax.set_ylabel(r"$\gamma^*$")
    ax.set_xlim(xmin=min(xmaxs), xmax=max(xmaxs))
    ax.set_ylim(ymin=gammast.mean - 0.25, ymax=gammast.mean + 0.25)
    ax.legend(loc="upper left")

    ax1 = plt.twinx(ax)
    chi2 = np.array([RESULTS[k]["chi2"] / (RESULTS[k]["Npts"]) for k in kall])
    ax1.plot(xmaxs, chi2, color="C1", label=r"$\frac{\chi^2}{dof}$", alpha=0.2)
    # ax1.set_ylim(ymin=0,ymax=2)
    ax1.legend(loc="upper right")
    ax1.set_ylabel(r"$\frac{\chi^2}{dof}$")
    ax1.set_title(r"$a\Omega_{{min}} = {{{0}}}$".format(f"{OMIN:.3f}"))

    # SECOND PLOT =====================================================
    ax2 = fig.add_subplot(2, 1, 2)

    OMAX = XMAXS[MA]

    # Scatter plot
    kin = [
        k
        for k in KEYS
        if float(k[1]) == OMAX
        if k in RESULTS
        if dGRAD[k] / CUTOFF <= NORM
    ]
    if FILTER is not None:
        kin = [k for k in kin if FILTER(RESULTS[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[0]) for k in kin]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kin])
    weight = np.array([AICw(RESULTS[k], weight="cut") for k in kin])
    ax2.scatter(xmaxs, gamma, c=mapp.to_rgba(weight))

    # Scatter out points
    kout = [
        k
        for k in KEYS
        if float(k[1]) == OMAX
        if k in RESULTS
        if not dGRAD[k] / CUTOFF <= NORM
    ]
    if FILTER is not None:
        kout = [k for k in kout if not FILTER(RESULTS[k]["pars"]["gamma"].mean)]
    xmaxs = [float(k[0]) for k in kout]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kout])
    weight = np.array([AICw(RESULTS[k], weight="cut") for k in kout])
    ax2.scatter(xmaxs, gamma, color="C0", alpha=0.1)

    # Error bands
    kall = [k for k in KEYS if float(k[1]) == OMAX if k in RESULTS]
    xmaxs = [float(k[0]) for k in kall]
    gamma = np.array([RESULTS[k]["pars"]["gamma"].mean for k in kall])
    errg = np.array([RESULTS[k]["pars"]["gamma"].sdev for k in kall])
    ax2.fill_between(xmaxs, gamma - errg, gamma + errg, color="gray", alpha=0.1)

    # Result span
    ax2.axhspan(
        gammast.mean + gammast.sdev,
        gammast.mean - gammast.sdev,
        alpha=0.2,
        label=r"$\gamma^*$".format(),
    )
    ax2.axhspan(gammast.mean + err_syst, gammast.mean - err_syst, alpha=0.2)

    ax2.set_xlabel(r"$a\Omega_{max}$")
    ax2.set_ylabel(r"$\gamma^*$")
    ax2.set_xlim(xmin=min(xmaxs), xmax=max(xmaxs))
    ax2.set_ylim(ymin=gammast.mean - 0.25, ymax=gammast.mean + 0.25)
    ax2.legend(loc="upper left")

    ax3 = plt.twinx(ax2)
    chi2 = np.array([RESULTS[k]["chi2"] / (RESULTS[k]["Npts"]) for k in kall])
    ax3.plot(xmaxs, chi2, color="C1", label=r"$\frac{\chi^2}{dof}$", alpha=0.2)
    ax3.set_ylim(ymin=0, ymax=2)
    ax3.legend(loc="upper right")
    ax3.set_ylabel(r"$\frac{\chi^2}{dof}$")
    ax3.set_title(r"$a\Omega_{{min}} = {{{0}}}$".format(f"{OMAX:.3f}"))

    fig.tight_layout()
    fig.savefig(out)


def do_modenumber_fit_aic(ensemble, filename, boot_gamma, plot_directory):
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

    RESULTS = WINDOWS(
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

    gstat, gsyst = WEIGHT(
        RESULTS,
        pars["norm"],
        pars["cutoff"],
        delta,
        weight="cut",
        FILTER=f(pars["filter"]),
    )

    print(gstat, gsyst)

    grad_plot(
        RESULTS,
        pars["norm"],
        pars["cutoff"],
        delta,
        FILTER=f(pars["filter"]),
        out=f"{plot_directory}/modenumber_grad_plot.pdf",
    )
    slice_plot(
        pars["plot_omega_min"],
        pars["plot_omega_max"],
        RESULTS,
        gsyst,
        gstat.sdev,
        delta,
        pars["norm"],
        pars["cutoff"],
        weight="cut",
        FILTER=f(pars["filter"]),
        out=f"{plot_directory}/modenumber_slice_plot.pdf",
    )

    if ensemble:
        add_measurement(ensemble["descriptor"], "gamma_aic", gstat.mean, gstat.sdev)
        add_measurement(ensemble["descriptor"], "gamma_aic_syst", gsyst.sdev)

    return gstat, gsyst
