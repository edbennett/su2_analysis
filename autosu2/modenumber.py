from collections import defaultdict
from re import compile
from argparse import ArgumentParser

import numpy as np
from numpy import array, asarray, isnan, mean, std
from numpy.random import randint, ranf
from sys import *
from scipy.optimize import curve_fit
from pandas import DataFrame
from numba import vectorize


CONFIGURATION_GETTER = compile(
    r'\[IO\]\[0\]Configuration \[.*n(?P<configuration>[0-9]+)'
)

def read_modenumber(filename):
    configuration = None
    modenumbers = defaultdict(dict)
    configuration_numbers = set()

    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('[IO][0]Configuration'):
                configuration = int(CONFIGURATION_GETTER.match(line)
                                    .group("configuration"))
                configuration_numbers.add(configuration)

            elif line.startswith('[MODENUMBER][0]nu['):
                split_line = line.split()
                omega = float(split_line[1])
                nu = float(split_line[4])
                assert configuration is not None
                modenumbers[omega][configuration] = nu

    for omega_modenumbers in modenumbers.values():
        assert len(omega_modenumbers) == len(configuration_numbers)

    return modenumbers


# a^{-4} \overline{\nu} \approx
#     a^{-4} \overline{\nu}_0 + A [(a\Lambda)^2 - (am)^2]^{\frac{2}{1+\gamma_*}}
@vectorize
def nubar_3param(lamb, A, am, gamma_star):
    return A * (lamb ** 2 - am ** 2) ** (2 / (1 + gamma_star))

@vectorize
def nubar_4param(lamb, A, am, gamma_star, nubar0):
    return nubar0 + nubar_3param(lamb, A, am, gamma_star)


# chi-square of function against data
def chisquare(func, indep, dep, sigma, params):
    chisquare = 0.0
    for point in range(len(indep)):
        chisquare += ((func(indep[point], *params) - dep[point]) ** 2
                      / sigma[point] ** 2)
    return chisquare / len(indep)
    
    
def old_fit(filename):
    DEBUG = True

    nus = read_modenumber(argv[1])
    lambs = sorted(nus.keys())

    # Pull an arbitrary element out of nus, since all have the same keys
    ns = list(next(iter(nus.values())).keys())
    lambs = sorted(lambs)
    n_confs = len(ns)
    n_lambs = len(lambs)
    n_array = array(ns)

    nubars = {}
    Snubars = {}
    nu_arrays = {}
    nu_estimates = {}

    minimal_window = None
    p_avgs = None

    #SET UP BASIC AVERAGE ARRAYS
    for lamb in lambs:
        nu_arrays[lamb] = array(list(nus[lamb].values()), dtype=np.float)
        nu_estimates[lamb] = []
        nubars[lamb] = mean(nu_arrays[lamb])
        nr = len(nu_arrays[lamb])
        for xx in range(1000):
            nu_estimates[lamb].append(
                mean(nu_arrays[lamb][randint(nr,size=nr)])
            )
        Snubars[lamb] = std(nu_estimates[lamb])
#        if DEBUG:
#            print(lamb, nubars[lamb], Snubars[lamb])

    min_chisquare = float("inf")
    #WINDOWING
    MIN_WINDOW_LENGTH = 6

    results = []

    for I in range(len(lambs)-MIN_WINDOW_LENGTH):
        for J in range(max(0,I+(MIN_WINDOW_LENGTH-1)),len(lambs)):
            lambs_window = lambs[I:J+1]

            p0 = [10000.0, 0.01, 1.0]

            success = False
            count = 0
            while not success:
                try:
                    #print(lambs_window)
                    #print([nubars[lamb] for lamb in lambs_window])
                    #print([Snubars[lamb] for lamb in lambs_window])
                    p1, p1sigma = curve_fit(
                        nubar_3param,
                        lambs_window,
                        [nubars[lamb] for lamb in lambs_window],
                        p0=[10000.0 * (1.0 + ranf()),
                            0.01 * (1 + ranf()),
                            ranf() * 0.1],
                        sigma=[Snubars[lamb] for lamb in lambs_window]
                    )
                    if DEBUG: 
                        pass
                        # print(p1)
                        # print(len(p1sigma), p1sigma)
                    # if (p1sigma == float("inf")
                    #     or p1sigma == float("-inf")
                    #     or p1sigma == float("NaN")):
                    #    raise Exception
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as ex:
                    print(ex)
                    count += 1
                    if count == 50:
                        print("Window", I, ",", J, "did not converge")
                        break
                else:
                    success = True
                if DEBUG:
                    if success:
                        print("Initial fit for window", I, ",", J,
                              "converged at ", p1)
            if not success:
                continue

            #NOW START BOOTSTRAPPING THIS WINDOW
            chisquareds = []
            ps = []
            for xx in range(4):
                ps.append([])

            badness = 0
            xx = 0
            niter = 0
            while xx < 1000:
                nubar_bs = []
                Snubar_bs = []
                n_set = n_array[randint(n_confs, size=2*n_confs)]
                #print(n_set)

                for lamb in lambs_window:
                    # print(nu_arrays[lamb])
                    nubar_bs.append(mean([nus[lamb][n] for n in n_set]))
                    Snubar_bs.append(std([nus[lamb][n] for n in n_set]))
                try:
                    p1_new = [p1[0] * (1.0 + (ranf() - 0.5)/10.0),
                              p1[1] * (1.0 + (ranf() - 0.5)/1.0),
                              p1[2] * (1.0 + (ranf() - 0.5)/10.0),
                              1000.0 * ranf()]
                    #if DEBUG:
                        # print("p1_new = ", p1_new)
                    p2, p2sigma = curve_fit(nubar_4param, lambs_window,
                                            nubar_bs,
                                            p0=p1_new, sigma=Snubar_bs)
                    #if DEBUG:
                        # print(p2, p2sigma)
                    if p2[3] < 0:
                        raise Exception
                    if abs(p2[1]) > lambs_window[0]:
                        badness = -1
                        break
                    if len(p2sigma) < 3:
                        raise Exception
                except (KeyboardInterrupt,SystemExit):
                    raise
                except Exception:
                    badness += 1
                    try:
                        p2
                    except NameError:
                        niter += 1
                        continue
                    if len(p2) == 0:
                        niter += 1
                        continue
                    if not float('-inf') < float(p2[0]) < float('inf'):
                        niter += 1
                        continue
                for yy in range(4):
                    ps[yy].append(p2[yy])
                # print(p2)
                chisquareds.append(chisquare(
                    nubar_4param, lambs_window, nubar_bs, Snubar_bs, tuple(p2)
                ))
                xx += 1
                niter += 1
                if niter == 10000:
                    print("Window", I, ",", J,
                          "giving up after 10,000 attempts")
                    break
            if badness == niter:
                print("Window", I, ",", J, "has 100% badness")
                continue
            if badness == -1:
                print("Window", I, ",", J, "breaks m lies within window")
                continue
            try:
                avg_chisquare = mean(chisquareds)
            except Exception:
                print("Caught exception in window", I, ",", J)
                continue
            p_avgs = []
            p_stds = []
            p_avgs.append(mean(ps[0]))
            p_stds.append(std(ps[0]))
            msquare = [elem**2 for elem in ps[1]]
            p_avgs.append(mean(msquare))
            p_stds.append(std(msquare))
            p_avgs.append(mean(ps[2]))
            p_stds.append(std(ps[2]))
            p_avgs.append(mean(ps[3]))
            p_stds.append(std(ps[3]))

            if avg_chisquare < min_chisquare:
                min_chisquare = avg_chisquare
                minimal_window = [I, J]
                minimal_params = p_avgs

            print("Window", I, ",", J,
                  "chisquare", avg_chisquare, "±", std(chisquareds),
                  ", parameters", p_avgs[0], "±", p_stds[0],
                  p_avgs[1], "±", p_stds[1],
                  p_avgs[2], "±", p_stds[2],
                  p_avgs[3], "±", p_stds[3],
                  "badness", badness/10.0, "%")

            results.append((
                lambs[I], lambs[J],
                avg_chisquare, std(chisquareds),
                p_avgs[0], p_stds[0],
                p_avgs[1], p_stds[1],
                p_avgs[2], p_stds[2],
                p_avgs[3], p_stds[3],
                badness / 10.0
            ))
                

    print("====================================================")
    print("Minimal chisquare of", min_chisquare, "at",
          minimal_window, "with fitted parameters", minimal_params)

    return DataFrame(results, columns=(
        'omega_lower_bound', 'omega_upper_bound',
        'chisquare', 'chisquare_error',
        'A', 'A_error', 'am', 'am_error',
        'gamma_star', 'gamma_star_error', 'nubar0', 'nubar0_error',
        'badness'
    ))



def main():
    parser = ArgumentParser()
    parser.add_argument('modenumber_filename')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    results = old_fit(args.modenumber_filename)
    if args.output is not None:
        results.to_csv(args.output)


if __name__ == '__main__':
    main()
