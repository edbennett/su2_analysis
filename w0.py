from numpy import argmax
from matplotlib.pyplot import subplots
from argparse import ArgumentParser
from bootstrap import basic_bootstrap, bootstrap_1d
from data import get_flows, write_results, get_output_filename


def ensemble_w0(times, Es, W0, ax=None, plot_label=None):
    h = times[1] - times[0]
    t2E = times ** 2 * Es
    tdt2Edt = times[1:-1] * (t2E[:, 2:] - t2E[:, :-2]) / (2 * h)
    if ax:
        tdt2Edt_avg, tdt2Edt_err = bootstrap_1d(tdt2Edt)
        ax.errorbar(
            times[1:-1], tdt2Edt_avg, yerr=tdt2Edt_err,
            fmt='.',
            label=plot_label
        )

    positions = argmax(tdt2Edt > W0, axis=1)
    W_positions_minus_one = tdt2Edt[list(zip(*enumerate(positions - 1)))]
    W_positions = tdt2Edt[list(zip(*enumerate(positions)))]
    w0_squared = times[positions] + h * (
        (W0 - W_positions_minus_one) /
        (W_positions - W_positions_minus_one)
    )
    return w0_squared ** 0.5


def main():
    parser = ArgumentParser()
    parser.add_argument('--filename', required=True)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--W0', default=0.35, type=float)
    parser.add_argument('--bootstrap_sample_count', default=200, type=int)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    if args.plot:
        fig, ax = subplots()
    else:
        fig, ax = None, None

    if not args.output_filename_prefix:
        args.output_filename_prefix = args.filename + '_'

    times, Eps, Ecs = get_flows(args.filename)
    w0ps = ensemble_w0(times, Eps, args.W0, ax=ax, plot_label=r'$w_0^p$')
    w0cs = ensemble_w0(times, Ecs, args.W0, ax=ax, plot_label=r'$w_0^c$')

    w0p = basic_bootstrap(
        w0ps,
        bootstrap_sample_count=args.bootstrap_sample_count
    )
    w0c = basic_bootstrap(
        w0cs,
        bootstrap_sample_count=args.bootstrap_sample_count
    )

    write_results(
        filename=get_output_filename(
            args.output_filename_prefix, 'w0', filetype='dat'
        ),
        headers=('w0p', 'w0p_error', 'w0c', 'w0c_error'),
        values_set=(w0p, w0c)
    )

    if args.plot:
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$t \frac{\mathrm{d}(t^2 E)}{\mathrm{d}t}$')
        ax.legend(loc=0, frameon=False)
        ax.axhline(args.W0, dashes=(1, 1))
        fig.tight_layout()
        fig.savefig(get_output_filename(
            args.output_filename_prefix,
            'W_flow',
            filetype='pdf'
        ))

    print(f"w0p: {w0p[0]} ± {w0p[1]}")
    print(f"w0c: {w0c[0]} ± {w0c[1]}")


if __name__ == '__main__':
    main()
