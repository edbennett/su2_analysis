#!/usr/bin/env python

from math import ceil
from numpy import nan

preliminary = False

beta_colour_marker = {
    1: [
        (2.05, "r", "o"),
        (2.1, "g", "x"),
        (2.15, "b", "^"),
        (2.2, "k", "3"),
        (2.25, "y", "*"),
        (2.3, "m", "4"),
        (2.4, "c", "v"),
    ],
    2: [
        (2.25, "y", "*"),
        (2.3, "m", "4"),
        (2.35, "darkorange", "+"),
    ],
}

bare_channel_labels = {
    "g5": r"\gamma_5",
    "gk": r"\gamma_k",
    "g5gk": r"\gamma_5\gamma_k",
    "id": "1",
    "A1++": "A^{++}",
    "E++": "E^{++}",
    "T2++": "T^{++}",
    "2++": "2^{++}",
    "spin12": r"\breve{g}",
    "sqrtsigma": r"\sqrt{\sigma}",
}

channel_labels = {
    "g5": r"$2^+$ scalar baryon",
    "gk": r"$0^-$ vector meson",
    "g5gk": r"$2^-$ vector baryon",
    "id": r"$2^-$ pseudoscalar baryon",
    "A1++": "$0^{++}$ scalar glueball",
    "2++": "$2^{++}$ tensor glueball",
    "spin12": r"$\breve{g}$ hybrid fermion",
    "sqrtsigma": r"$\sqrt{\sigma}$ string tension",
}

figlegend_defaults = {
    "loc": "outside lower center",
    "columnspacing": 0.9,
    "handletextpad": 0,
}

ONE_COLUMN = 3.5
TWO_COLUMN = 7.0

# TODO: get the numbers for the critical regions programmatically
critical_ms = {
    1: [-1.5256, -1.4760, -1.4289, -1.3932, None, None, None],
    2: [None, None, None],
}


def add_figure_key(fig, markers=True, Nfs=[1], nrow=1, shortlabel=True):
    legend_contents = [
        fig.axes[0].errorbar(
            [nan],
            [nan],
            yerr=[nan],
            xerr=[nan],
            color=colour,
            marker=f"{marker if markers else ','}",
            label=f"${beta}$" if shortlabel else f"$\\beta={beta}$",
            linestyle="",
        )
        for beta, colour, marker in sorted(
            set(bcm for Nf in Nfs for bcm in beta_colour_marker[Nf])
        )
    ]
    fig.legend(
        handles=legend_contents,
        ncol=ceil(8 / nrow),
        title=r"$\beta$" if shortlabel else None,
        **figlegend_defaults,
    )


def format_ensembles_list(ensemble_names):
    if len(ensemble_names) > 2:
        return "{}, and {}".format(", ".join(ensemble_names[:-1]), ensemble_names[-1])
    elif len(ensemble_names) == 2:
        return "{} and {}".format(*ensemble_names)
    elif len(ensemble_names) == 1:
        return "{}".format(*ensemble_names)
    return ""


def generate(*args, **kwargs):
    pass
