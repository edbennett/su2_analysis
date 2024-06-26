import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from ..derived_observables import merge_quantities
from ..plots import set_plot_defaults

from .common import beta_colour_marker


def fit_form(x, a, b, c):
    return a + b * x + c * x**2


def get_subset(data, Nf, beta):
    merged_data = merge_quantities(data, ["mpcac_mass"]).dropna(
        subset=["value_w0", "value_mpcac_mass"]
    )
    return merged_data[
        (merged_data.Nf == Nf)
        & (merged_data.beta == beta)
        & ~(merged_data.label.str.endswith("*"))
    ]


def fit_1_over_w0(data, Nf, beta):
    subset = get_subset(data, Nf, beta)
    if len(subset) == 0:
        return np.nan * np.ones(3), np.nan * np.ones((3, 3))

    return curve_fit(
        fit_form,
        subset.value_mpcac_mass,
        1 / subset.value_w0,
        sigma=subset.uncertainty_w0 / subset.value_w0**2,
    )


def generate(data, ensembles):
    set_plot_defaults()

    filename = "assets/plots/w0_chiral.pdf"

    fig, axes = plt.subplots(
        ncols=2, layout="constrained", figsize=(7, 3.5), sharey=True
    )

    for ax, (Nf, betas) in zip(axes, beta_colour_marker.items()):
        for beta, colour, marker in betas:
            subset = get_subset(data, Nf, beta)
            if len(subset) == 0:
                continue
            ax.errorbar(
                subset.value_mpcac_mass,
                1 / subset.value_w0,
                xerr=subset.uncertainty_mpcac_mass,
                yerr=subset.uncertainty_w0 / subset.value_w0**2,
                color=colour,
                marker=marker,
                ls="none",
                label=f"{beta}",
            )

            fit_values, _ = fit_1_over_w0(data, Nf, beta)
            x_range = np.linspace(0, max(subset.value_mpcac_mass) * 1.05, 1000)
            ax.plot(
                x_range,
                fit_form(x_range, *fit_values),
                color=colour,
                alpha=0.5,
                dashes=(2, 2),
            )

        ax.set_title(f"$N_{{\\mathrm{{f}}}} = {Nf}$")
        ax.set_xlabel(r"$am_{{\mathrm{{PCAC}}}}$")

    axes[0].set_ylabel(r"$a / w_0$")
    fig.legend(loc="outside lower center", title=r"$\beta$", ncols=8)

    fig.savefig(filename)
