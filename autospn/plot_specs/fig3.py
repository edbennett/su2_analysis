from shutil import copyfile

from ..db import measurement_is_up_to_date, get_measurement_as_ufloat

CAPTION = r'''
Topological charge histories (left), and histograms (right), for the ensembles
 DB1M1, DB3M5, and DB5M1, respectively. Fitted parameters are
(a) $Q_0={DB1M1_Q0:.2uSL}$, $\sigma={DB1M1_width:.2uSL}$;
(b) $Q_0={DB3M5_Q0:.2uSL}$, $\sigma={DB3M5_width:.2uSL}$;
(c) $Q_0={DB5M1_Q0:.2uSL}$, $\sigma={DB5M1_width:.2uSL}$.'''


def generate(data):
    assert measurement_is_up_to_date(
        {'label': 'DB1M1'}, 'fitted_Q0',
        compare_file='raw_data/nf2_FUN/32x16x16x16b6.9m-0.85/out_wflow'
    )
    assert measurement_is_up_to_date(
        {'label': 'DB3M5'}, 'fitted_Q0',
        compare_file='raw_data/nf2_FUN/36x24x24x24b7.2m-0.78/out_wflow'
    )
    assert measurement_is_up_to_date(
        {'label': 'DB5M1'}, 'fitted_Q0',
        compare_file='raw_data/nf2_FUN/48x24x24x24b7.5m-0.69/out_wflow'
    )

    copyfile('processed_data/nf2_FUN/32x16x16x16b6.9m-0.85/Q.pdf',
             'final_plots/fig3a.pdf')
    copyfile('processed_data/nf2_FUN/36x24x24x24b7.2m-0.78/Q.pdf',
             'final_plots/fig3b.pdf')
    copyfile('processed_data/nf2_FUN/48x24x24x24b7.5m-0.69/Q.pdf',
             'final_plots/fig3c.pdf')

    ensembles = [{'label': 'DB1M1'}, {'label': 'DB3M5'}, {'label': 'DB5M1'}]
    observables = {}
    for ensemble in ensembles:
        observables[f'{ensemble["label"]}_Q0'] = get_measurement_as_ufloat(
            ensemble, 'fitted_Q0'
        )
        observables[f'{ensemble["label"]}_width'] = get_measurement_as_ufloat(
            ensemble, 'Q_width'
        )

    with open('final_plots/fig3.tex', 'w') as f:
        print(r'\begin{figure}', file=f)
        print(r'  \center', file=f)
        print(r'\captionsetup[subfigure]{aboveskip=-10pt}', file=f)
        for filename, caption in (
                ('fig3a', r'$\beta=6.9,m=-0.85,V=32\times16^3$'),
                ('fig3b', r'$\beta=7.2,m=-0.78,V=36\times24^3$'),
                ('fig3c', r'$\beta=7.5,m=-0.69,V=48\times24^3$')):
            print(r'  \begin{subfigure}{\textwidth}', file=f)
            print(r'    \center', file=f)
            print(r'    \includegraphics[width=0.9\textwidth]{{{fn}}}'.format(
                fn=filename
            ), file=f)
            print(r'    \caption{{{caption}}}'.format(caption=caption),
                  file=f)
            print(r'  \end{subfigure}', file=f)

        caption = CAPTION.format(**observables)
        print(r'  \caption{{{caption}}}'.format(caption=caption), file=f)
        print(r'  \label{fig:topcharge}', file=f)
        print(r'\end{figure}', file=f)
