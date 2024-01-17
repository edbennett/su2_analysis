import gvar as gv
import lsqfit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from uncertainties import ufloat

from .common import preliminary
from ..plots import set_plot_defaults
from ..tables import generate_table_from_content, table_row
from ..derived_observables import merge_no_w0


def quadratic(x, a, b, c):
    """
    A simple quadratic, to keep the fit consistent with the plot
    """
    return a + b * x + c * x**2


class Quadratic(lsqfit.MultiFitterModel):
    def __init__(self, datatag, m, L, c0, gammam, omega, alpha0, alpha1, alpha2, sm):
        super().__init__(datatag)
        self.m = np.array(m)
        self.L = np.array(L)

        # keys used to find the intercept and slope in a parameter dictionary
        self.c0 = c0
        self.gammam = gammam
        self.omega = omega
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.sm = sm

    def fitfcn(self, p):
        if isinstance(self.gammam, str):
            gammam = p[self.gammam]
        else:
            gamma = self.gammam

        scaled_m = p[self.sm] * self.m
        x = self.L * scaled_m ** (1 / (1 + gammam))
        f_x = p[self.alpha0] + p[self.alpha1] * x + p[self.alpha2] * x**2
        return f_x * (1 + p[self.c0] * scaled_m ** p[self.omega])

    def buildprior(self, prior, mopt=None):
        mprior = gv.BufferDict()
        model_keys = [
            self.c0,
            self.gammam,
            self.omega,
            self.alpha0,
            self.alpha1,
            self.alpha2,
            self.sm,
        ]
        for k in gv.get_dictkeys(prior, model_keys):
            mprior[k] = prior[k]
        return mprior

    def builddata(self, data):
        "Extract the model's fit data from data."
        return data[self.datatag]


def fit(merged_data, betas_to_fit, universal_fit=True, gamma_star=None):
    models = []
    data_to_fit = {}
    states = "g5", "gk"
    for state in states:
        if universal_fit:
            gammam, omega = "gammam", "omega"
        else:
            gammam, omega = f"gammam_{state}", f"omega_{state}"

        for beta in betas_to_fit:
            subset = merged_data[merged_data.beta == beta].dropna(
                subset=[f"value_{state}_mass", f"uncertainty_{state}_mass"]
            )
            L = subset.L.values
            mpcac = subset.value_mpcac_mass.values

            models.append(
                Quadratic(
                    f"{state}_{beta}",
                    m=mpcac,
                    L=L,
                    c0=f"c0_{state}",
                    gammam=gammam,
                    omega=omega,
                    alpha0=f"alpha0_{state}",
                    alpha1=f"alpha1_{state}",
                    alpha2=f"alpha2_{state}",
                    sm=f"sm_b{beta}",
                )
            )
            data_to_fit[f"{state}_{beta}"] = gv.gvar(
                (L * subset[f"value_{state}_mass"]).values,
                np.diag(L * subset[f"uncertainty_{state}_mass"].values) ** 2,
            )

    if universal_fit:
        priors = {"gammam": "0.0(1.0)", "omega": "-0.1(1.0)"}
    else:
        priors = {}
        priors.update([(f"gammam_{state}", "0.0(1.0)") for state in states])
        priors.update([(f"omega_{state}", "-0.1(1.0)") for state in states])
    priors.update([(f"c0_{state}", "0.0(1.0)") for state in states])
    priors.update(
        [(f"alpha{idx}_{state}", "-0.1(10.0)") for state in states for idx in (0, 1, 2)]
    )
    priors.update([(f"sm_b{beta}", "1(1)") for beta in betas_to_fit])

    fitter = lsqfit.MultiFitter(models=models)
    fit_result = fitter.lsqfit(data=gv.gvar(data_to_fit), prior=priors, maxit=10_000)
    return fit_result


def plot(data, fit_result, betas_to_fit, filename=None):
    set_plot_defaults(preliminary=preliminary)
    fig, ax = plt.subplots()
    states = "g5", "gk"
    markers = {"g5": ".", "gk": "x"}
    labels = {"g5": r"\gamma_5", "gk": r"\gamma_k"}

    # gammam = fit_result.p["gammam"]
    # omega = fit_result.p["omega"]
    if "omega" in fit_result.p:
        universal = True
        gammam = fit_result.p["gammam"]
        omega = fit_result.p["omega"]
    else:
        universal = False

    for beta_index, beta in enumerate(betas_to_fit):
        subset_data = data[data.beta == beta]
        L = subset_data.L
        ax.errorbar(
            [np.nan],
            [np.nan],
            yerr=[np.nan],
            label=r"$\beta=" f"{beta}$",
            ls="none",
            marker=".",
        )

        for state in states:
            c0 = fit_result.p[f"c0_{state}"].mean
            sm = fit_result.p[f"sm_b{beta}"].mean
            if not universal:
                omega = fit_result.p[f"omega_{state}"]
                gammam = fit_result.p[f"gammam_{state}"]
            scaled_m = sm * subset_data.value_mpcac_mass

            ax.errorbar(
                L * scaled_m ** (1 / (1 + gammam.mean)),
                L
                * subset_data[f"value_{state}_mass"]
                / (1 + c0 * scaled_m**omega.mean),
                yerr=L
                * subset_data.uncertainty_g5_mass
                / (1 + c0 * scaled_m**omega.mean),
                color=f"C{beta_index}",
                marker=markers[state],
                ls="none",
            )

    ax.set_xlabel(r"$L(s_m m_{\mathrm{PCAC}})^{1/(1+\gamma_*)}$")
    ax.set_ylabel(r"$LM_{H}/(1+c_0(s_m m)^\omega)$")
    ax.set_xlim((0, None))
    ax.set_ylim((0, None))

    xlim = ax.get_xlim()
    xrange = np.linspace(*xlim, 1000)
    for state in states:
        alpha0 = fit_result.p[f"alpha0_{state}"].mean
        alpha1 = fit_result.p[f"alpha1_{state}"].mean
        alpha2 = fit_result.p[f"alpha2_{state}"].mean
        slug = ""
        if not universal:
            gammam = fit_result.p[f"gammam_{state}"]
            omega = fit_result.p[f"omega_{state}"]
            slug = f"\\Rightarrow\\gamma_m={gammam},\\omega={omega}"
        ax.plot(
            xrange,
            quadratic(xrange, alpha0, alpha1, alpha2),
            label=f"${labels[state]}$: ${alpha0:.3f}+{alpha1:.3f}x+{alpha2:.3f}x^2{slug}$",
        )

    if universal:
        title = (
            r"$\gamma_*="
            f"{ufloat(gammam.mean, gammam.sdev):.2uS}$, "
            r"$\omega="
            f"{ufloat(omega.mean, omega.sdev):.2uS}$"
        )
    else:
        title = ""
    title = (
        f"$N_f={set(data.Nf).pop()}$, "
        + title
        + (", " if title else "")
        + f"$\\chi^2/$d.o.f = {fit_result.chi2 / fit_result.dof:.2e}"
    )
    ax.set_title(title)

    ax.legend(frameon=False)

    if filename is None:
        plt.show()
    else:
        if filename != "":
            fig.savefig(filename)
        plt.close(fig)


def tabulate(fit_results, filename):
    columns = [
        r"$\beta_{\mathrm{min}}$",
        None,
        r"$\gamma_*$",
        r"$\omega$",
        None,
        r"$y_{\mathrm{m}}$",
        r"$y_0$",
        None,
        r"$\alpha_0$",
        r"$\alpha_1$",
        r"$\alpha_2$",
        None,
        r"$c_{\beta}$",
        None,
        r"$\chi^2/\mathrm{d.o.f.}$",
    ]
    table_content = []
    add_hline = False
    num_format = r"${}$"  # r'${:.2uSL}$'
    for betas, fit_result in fit_results:
        if betas is None:
            add_hline = True
            continue

        p = fit_result.p
        ym = p["ym"]
        gamma_star = ym - 1
        if isinstance(gamma_star, float):
            gamma_star_text = f"{gamma_star:.2f}"
            ym_text = f"{ym:.2f}"
        else:
            gamma_star_text = f"{gamma_star}"
            ym_text = f"{ym}"

        omega = -p["y0"] / p["ym"]

        row_content = [
            num_format.format(value)
            for value in (
                min(betas),
                gamma_star_text,
                omega,
                ym_text,
                p["y0"],
                p["alpha0"],
                p["alpha1"],
                p["alpha2"],
            )
        ] + [
            "$" + ", ".join(map(str, p["c"])) + "$",
            f"{fit_result.chi2 / fit_result.dof:.2f}",
        ]
        table_content.append(
            ("    \\hline\n" if add_hline else "") + table_row(row_content)
        )
        add_hline = False

    generate_table_from_content(filename, table_content, columns=columns)


def generate_single_Nf(Nf, betas_to_fit_set, merged_data, ensembles):
    filename = f"{{}}_plots/Nf{Nf}_nearmarginal_{{}}beta{{}}.pdf"
    table_filename = f"Nf{Nf}_nearmarginal{{}}.tex"
    fit_results_free = []
    fit_results_constrained = []

    for universal_fit in True, False:
        for betas_to_fit in betas_to_fit_set:
            fit_result = fit(merged_data, betas_to_fit, universal_fit=universal_fit)
            print(f"{universal_fit=}, {betas_to_fit=}")
            print("===")
            print(fit_result.format())
            print()
            fit_results_free.append((betas_to_fit, fit_result))
            plot(
                merged_data,
                fit_result,
                betas_to_fit,
                filename.format(
                    "final",
                    len(betas_to_fit),
                    "_universal" if universal_fit else "_nonuniversal",
                ),
            )


def generate(data, ensembles):
    merged_data = {
        Nf: merge_no_w0(
            data[data.Nf == Nf], ["g5_mass", "gk_mass", "mpcac_mass"]
        ).dropna(subset=("value_mpcac_mass", "value_g5_mass", "value_gk_mass"))
        for Nf in (1, 2)
    }
    betas_to_fit_set = {
        1: [
            (2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.4),
            (2.1, 2.15, 2.2, 2.25, 2.3, 2.4),
            (2.1, 2.15, 2.2, 2.25, 2.3),
            (2.25, 2.3, 2.4),
            (2.25, 2.3),
        ],
        2: [(2.25, 2.35)],
    }

    extra_data_Nf2 = pd.read_csv("external_data/su2_nf2_b2.25.csv")
    merged_data[2] = pd.concat([merged_data[2], extra_data_Nf2], ignore_index=True)

    for Nf in 1, 2:
        generate_single_Nf(Nf, betas_to_fit_set[Nf], merged_data[Nf], ensembles)
