#!/usr/bin/env python

from numpy import nan

beta_colour_marker = [
    (2.05, 'r', 'o'),
    (2.1, 'g', 'x'),
    (2.15, 'b', '^'),
    (2.2, 'k', '3')
]

bare_channel_labels = {
    'g5': r'\gamma_5',
    'gk': r'\gamma_k',
    'g5gk': r'\gamma_5\gamma_k',
    'id': '1',
    'App': 'A^{++}',
    'Epp': 'E^{++}',
    'Tpp': 'T^{++}',
    'spin12': r'\breve{g}',
    'sqrtsigma': r'\sqrt{\sigma}',
}

channel_labels = {
    'g5': r'$2^+$ scalar baryon',
    'gk': r'$0^-$ vector meson',
    'g5gk': r'$2^-$ vector baryon',
    'id': r'$2^-$ pseudoscalar baryon',
    'App': '$0^{++}$ scalar glueball',
    'Epp': '$2^{++}$ tensor glueball',
    'Tpp': '$2^{++}$ tensor glueball',
    'spin12': r'$\breve{g}$',
    'sqrtsigma': r'$\sqrt{\sigma}$',
}

# TODO: get the numberes for the critical regions programmatically
critical_ms = [-1.5256, -1.4760, -1.4289, -1.3932]


def add_figure_key(fig, markers=True):
    legend_contents = [
        fig.axes[0].errorbar(
            [-1], [-1], yerr=[nan], xerr=[nan],
            color=colour,
            marker=f'{marker if markers else ","}',
            label=f"$\\beta={beta}$")
        for beta, colour, marker in beta_colour_marker
    ]

    fig.legend(
        handles=legend_contents,
        loc='upper center',
        frameon=False,
        ncol=4,
        columnspacing=1.0,
        handletextpad=0,
        borderpad=0
    )


def generate(*args, **kwargs):
    pass
