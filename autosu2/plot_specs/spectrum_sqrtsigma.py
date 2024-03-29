from numpy import nan
import matplotlib.pyplot as plt

from ..plots import set_plot_defaults
from ..derived_observables import merge_quantities

from .common import beta_colour_marker, critical_ms, preliminary


def generate(data, ensembles):
    set_plot_defaults(markersize=3, capsize=0.5, linewidth=0.3, preliminary=preliminary)

    use_pcac = True

    filename = "auxiliary_plots/spectrum_sqrtsigma_Nf{Nf}.pdf"
    quantities = (
        "mpcac_mass",
        "g5_mass",
        "g5_decay_const",
        "gk_mass",
        "gk_decay_const" "g5gk_mass",
        "id_mass",
        "A1++_mass",
        "E++_mass",
        "T2++_mass",
        "spin12_mass",
    )

    merged_data = merge_quantities(data, quantities + ("sqrtsigma",))
    for quantity in quantities:
        merged_data[f"value_{quantity}_over_sqrtsigma"] = (
            merged_data[f"value_{quantity}"] / merged_data["value_sqrtsigma"]
        )
        merged_data[f"uncertainty_{quantity}_over_sqrtsigma"] = (
            merged_data[f"uncertainty_{quantity}"] ** 2
            / merged_data["value_sqrtsigma"] ** 2
            + merged_data["uncertainty_sqrtsigma"] ** 2
            * merged_data[f"value_{quantity}"] ** 2
            / merged_data["value_sqrtsigma"] ** 4
        ) ** 0.5

    channels_to_plot = "g5", "g5gk", "id", "A1++", "E++", "T2++", "spin12"
    channel_labels = (
        r"\gamma_5",
        r"\gamma_5\gamma_k",
        "1",
        "A^{++}",
        "E^{++}",
        "T^{++}",
        r"\breve{g}",
    )
    markers = ".", "x", "+", "^", "v", "1", "*"

    for Nf in 1, 2:
        fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(4.5, 8))
        if use_pcac:
            axes[-1].set_xlabel(r"$m_{\mathrm{PCAC}} / \sqrt{\sigma}$")
        else:
            axes[-1].set_xlabel(r"$(m - m_c) / \sqrt{\sigma}$")

        axes[0].set_ylabel(r"$M / \sqrt{\sigma}$")
        axes[1].set_ylabel(r"$M / \sqrt{\sigma}$")
        axes[2].set_ylabel(r"$f / \sqrt{\sigma}$")

        for (beta, colour, _), m_c in zip(beta_colour_marker[Nf], critical_ms[Nf]):
            data_to_plot = merged_data[
                (merged_data.beta == beta)
                & ~(merged_data.label.str.endswith("*"))
                & (merged_data.Nf == Nf)
            ]
            if use_pcac:
                m_over_sqrtsigma = data_to_plot.value_mpcac_mass_over_sqrtsigma
                m_over_sqrtsigma_err = (
                    data_to_plot.uncertainty_mpcac_mass_over_sqrtsigma
                )
            else:
                m_over_sqrtsigma = (data_to_plot.m - m_c) / data_to_plot.value_sqrtsigma
                m_over_sqrtsigma_err = (
                    (data_to_plot.m - m_c)
                    * data_to_plot.uncertainty_sqrtsigma
                    / data_to_plot.value_sqrtsigma**2
                )

            for observable in "mass", "decay_const":
                for channel, marker in zip(channels_to_plot, markers):
                    if observable == "decay_const":
                        ax = axes[2]
                    elif channel in ("id", "g5", "g5gk", "gk"):
                        ax = axes[0]
                    else:
                        ax = axes[1]
                    suffix = f"{channel}_{observable}_over_sqrtsigma"
                    if f"value_{suffix}" in data_to_plot:
                        ax.errorbar(
                            m_over_sqrtsigma,
                            data_to_plot[f"value_{suffix}"],
                            xerr=m_over_sqrtsigma_err,
                            yerr=data_to_plot[f"uncertainty_{suffix}"],
                            color=colour,
                            marker=marker,
                            linestyle="none",
                        )

            axes[0].errorbar(
                [-1],
                [-1],
                yerr=[nan],
                xerr=[nan],
                color=colour,
                marker=",",
                label=f"$\\beta={beta}$",
            )

        for channel_label, marker in zip(channel_labels, markers):
            axes[0].scatter(
                [-1], [-1], marker=marker, color="black", label=f"${channel_label}$"
            )

        axes[0].legend(
            loc="lower right",
            frameon=False,
            ncol=4,
            columnspacing=1.0,
            bbox_to_anchor=(0, 1.02, 1, 0.25),
        )

        axes[0].set_xlim((0, None))
        for ax in axes:
            ax.set_ylim((0, None))

        fig.tight_layout()
        fig.savefig(filename.format(Nf=Nf))
        plt.close(fig)
