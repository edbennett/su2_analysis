from re import findall

from matplotlib.pyplot import show, subplots
from numpy import arange, asarray, exp, histogram, inf, linspace, nan, ones
from scipy.optimize import curve_fit, minimize

from .db import measurement_is_up_to_date, add_measurement
from .data import get_filename
from .plots import set_plot_defaults

SMALL_LOWER_BOUND = 1e-16
MAX_IMAG_PART = 1e-16

PLAQ_TO_IDX = {(1, 0): 0,
               (2, 0): 1,
               (3, 0): 2,
               (2, 1): 3,
               (3, 1): 4,
               (3, 2): 5}


def get_loops_from_raw(filename, first_config):
    configs = []
    plaqs = [[] for _ in range(6)]
    loops = [[] for _ in range(4)]

    for line in open(filename):
        line_contents = line.split()
        if (line_contents[0] == '[IO][0]Configuration' 
                and line_contents[2] == 'read'):
            for observable_set in (plaqs + loops):
                if len(observable_set) != len(configs):
                    import pdb; pdb.set_trace()
                assert len(observable_set) == len(configs)

            configs.append(int(findall(r'.*n(\d+)(?:_[0-9.]+)?]',
                                       line_contents[1])[0]))
            continue

        if line_contents[0] == '[PLAQ][0]Plaq(':
            dir1 = int(line_contents[1])
            dir2 = int(line_contents[3].replace(')', ''))
            plaqs[PLAQ_TO_IDX[(dir1, dir2)]].append(
                float(line_contents[6]) + float(line_contents[8]) * 1J
            )
        elif line_contents[0].startswith('[PLAQ][0]Plaq('):
            directions = findall(r'\[PLAQ\]\[0\]Plaq\((\d),(\d)\)',
                                 line_contents[0])[0]
            dir1 = int(directions[0])
            dir2 = int(directions[1])
            plaqs[PLAQ_TO_IDX[(dir1, dir2)]].append(
                float(line_contents[3]) + float(line_contents[5]) * 1J
            )            
        elif line_contents[0] in ('[POLYAKOV][0]Polyakov',
                                  '[FUND_POLYAKOV][0]Polyakov'):
            dir1 = int(line_contents[2])
            loops[dir1].append(
                float(line_contents[4]) + float(line_contents[5]) * 1J
            )

    configs_array = asarray(configs)

    return (configs_array,
            [asarray(plaq)[configs_array >= first_config] for plaq in plaqs],
            [asarray(loop)[configs_array >= first_config] for loop in loops])


def plot_loops(loop_histograms,
               filename=None, ax=None, fitted_params=None, fit_form=None):
    assert not (filename and ax)

    set_plot_defaults()

    if not ax:
        received_ax = False
        fig, ax = subplots()
    else:
        received_ax = True

    ax.set_xlabel(r'$\langle P_\mu\rangle$')
    ax.set_ylabel(r'$N$')

    bins, histograms = loop_histograms

    for direction, histogram in enumerate(histograms):
        ax.plot(bins, histogram, drawstyle='steps-pre', label=f'{direction}')

    if fitted_params and fit_form:
        x_values = linspace(min(bins), max(bins), 1000)
        for direction, fitted_param in enumerate(fitted_params):
            ax.plot(x_values, fit_form(x_values, *fitted_param),
                    color=f'C{direction}')
    elif fitted_params:
        print("WARNING: fitted_params specified but no fit form, "
              "not plotting a fit.")
    elif fit_form:
        print("WARNING: fit_form specified but no fitted_params, "
              "not plotting a fit.")

    ax.set_ylim((0, None))
    ax.legend(loc=0, frameon=False, title=r'$\mu$')

    if not received_ax:
        fig.tight_layout()
        if filename:
            fig.savefig(filename)
        else:
            show()


def fit_loops(loop_histograms, fit_form):
    bins, histograms = loop_histograms
    bin_centres = (bins[:-1] + bins[1:]) / 2

    fitted_params = []
    for histogram in histograms:
        p0 = (histogram.max(), SMALL_LOWER_BOUND, bins.mean())
        
        popt, pcov = curve_fit_with_minimize(
            fit_form, bin_centres, histogram,
            sigma=(histogram + 1) ** 0.5, absolute_sigma=False, p0=p0,
            bounds=((0.1 * histogram.max(), 0, SMALL_LOWER_BOUND),
                    (10 * histogram.max(), bins.max(), bins.max())),
            # method='trf'
        )
        fitted_params.append(list(zip(popt, pcov.diagonal())))

    return fitted_params


def mirror_gaussian_fit_form(x, amplitude, centre, sigma):
    return amplitude * (exp(-(x - centre) ** 2 / (2 * sigma))
                        + exp(-(x + centre) ** 2 / (2 * sigma))) / 2


def gaussian_fit_form(x, amplitude, centre, sigma):
    return amplitude * exp(-(x - centre) ** 2 / (2 * sigma))


def get_bins(loops, bin_width=None, num_bins=None):
    if (not bin_width and not num_bins) or (bin_width and num_bins):
        raise ValueError(
            "Exactly one of `bin_width` and `num_bins` should be specified"
        )
    loops_array = asarray(loops)
    min_loop = min(0, loops_array.min())
    max_loop = loops_array.max()

    if num_bins:
        return linspace(min_loop, max_loop, num_bins + 1)
    else:
        return arange(min_loop, max_loop + bin_width, bin_width)


def curve_fit_with_minimize(f, xdata, ydata, p0,
                            sigma=None, absolute_sigma=False, bounds=None):
    if bounds is not None:
        min_bounds = tuple(zip(*bounds))
    else:
        min_bounds = None

    if sigma is not None:
        def fun(p):
            return ((f(xdata, *p) - ydata) ** 2 / sigma ** 2).sum()
    else:
        def fun(p):
            return ((f(xdata, *p) - ydata) ** 2).sum()

    result = minimize(fun, p0, bounds=min_bounds, method='trust-constr')

    return result.x, ones((len(p0), len(p0))) * nan


def fit_and_plot_polyakov_loops(filename,
                                bin_width=None, num_bins=None, first_config=0,
                                plot_filename=None, ax=None, do_fit=True):
    configs, plaqs, loops = get_loops_from_raw(filename, first_config)

    for observable in plaqs + loops:
        assert observable.mean().imag < MAX_IMAG_PART

    bins = get_bins(loops, bin_width, num_bins)
    histogrammed_loops = [
        histogram(loop.real, bins=bins)[0] for loop in loops
    ]
    bin_lower_edges = bins[:-1]

    if do_fit:
        fitted_params_with_errors = fit_loops(
            (bins, histogrammed_loops), mirror_gaussian_fit_form
        )
        fitted_params = [tuple(param[0] for param in param_set)
                         for param_set in fitted_params_with_errors]
        fit_form = gaussian_fit_form
    else:
        fitted_params_with_errors = None
        fitted_params = None
        fit_form = None

    plot_loops((bin_lower_edges, histogrammed_loops),
               filename=plot_filename,
               fitted_params=fitted_params,
               fit_form=fit_form)

    return fitted_params_with_errors


def fit_plot_and_save_polyakov_loops(simulation_descriptor=None,
                                     filename_formatter=None,
                                     filename=None,
                                     plot_filename_formatter=None,
                                     plot_filename=None,
                                     do_fit=True,
                                     force=False):
    filename = get_filename(simulation_descriptor,
                            filename_formatter, filename)
    plot_filename = get_filename(simulation_descriptor,
                                 plot_filename_formatter,
                                 plot_filename,
                                 optional=True)

    if (simulation_descriptor
        and not force
        and (measurement_is_up_to_date(
            simulation_descriptor, 'polyakov_0_centre',
            compare_file=filename)
        )
    ):
        # Already up to date
        return

    if simulation_descriptor and 'first_cfg' in simulation_descriptor:
        first_config = simulation_descriptor['first_cfg']
    else:
        first_config = 0

    fit_results = fit_and_plot_polyakov_loops(
        filename,
        plot_filename=plot_filename,
        num_bins=50,
        first_config=0,
        do_fit=do_fit
    )

    for direction, fit_result_set in enumerate(fit_results):
        for observable, result in (
                zip(('amplitude', 'centre', 'sigma'), fit_result_set)
        ):
            add_measurement(simulation_descriptor,
                            f'polyakov_{direction}_{observable}',
                            *result)
    return fit_results


def main():
    from argparse import ArgumentParser
    from pprint import pprint

    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--plot_filename', default=None)
    parser.add_argument('--bin_width', default=None, type=float)
    parser.add_argument('--num_bins', default=None, type=int)
    parser.add_argument('--skip_fit', dest='do_fit', action='store_false')
    args = parser.parse_args()

    pprint(fit_and_plot_polyakov_loops(**vars(args)))


if __name__ == '__main__':
    main()

