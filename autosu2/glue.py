#!/usr/bin/env python

from itertools import product
from numpy import dtype, empty
from scipy.io import FortranFile


DEFAULT_GLUE_PARAMS = {
    "num_operators_bctn": 6,
    "num_operators_L": 1,
    "num_operators_L2": 2,
    "num_operators_0": 2,
    "num_operators_E": 6,
    "num_operators_T": 3,
}


def read_glue_correlation_matrices(
    filename,
    num_points_T,
    num_configs,
    num_bins,
    num_momenta,
    num_blocking_levels,
    glue_params=DEFAULT_GLUE_PARAMS,
):
    """
    Reads binned, jackknifed samples of gluonic correlation matrices,
    as output in Fortran binary format by the Fortran glueball code.
    """
    # Other than torelons, states that are larger than half the lattice
    # extent double-count and so are omitted.
    L_max = num_points_T // 2 + 1

    # Open the file and read header; 4-byte little-endian unsigned
    with FortranFile(filename, "r", "<u4") as ff:
        bctn = ff.read_record(
            dtype(("<f8", (glue_params["num_operators_bctn"], num_bins)))
        )
        cor_L, vac_L = ff.read_record(
            dtype(
                (
                    "<c16",
                    (
                        num_blocking_levels,
                        num_blocking_levels,
                        L_max,
                        glue_params["num_operators_L"],
                        glue_params["num_operators_L"],
                        num_momenta,
                        num_bins,
                    ),
                )
            ),
            dtype(
                ("<f8", (num_blocking_levels, glue_params["num_operators_L"], num_bins))
            ),
        )
        cor_L2, vac_L2 = ff.read_record(
            dtype(
                (
                    "<c16",
                    (
                        num_blocking_levels,
                        num_blocking_levels,
                        L_max,
                        glue_params["num_operators_L2"],
                        glue_params["num_operators_L2"],
                        num_momenta,
                        num_bins,
                    ),
                )
            ),
            dtype(
                (
                    "<f8",
                    (num_blocking_levels, glue_params["num_operators_L2"], num_bins),
                )
            ),
        )
        cor_0R, vac_0, cor_ER, cor_TR = ff.read_record(
            dtype(
                (
                    "<f8",
                    (
                        num_blocking_levels,
                        num_blocking_levels,
                        L_max,
                        glue_params["num_operators_0"],
                        glue_params["num_operators_0"],
                        num_bins,
                    ),
                )
            ),
            dtype(
                ("<f8", (num_blocking_levels, glue_params["num_operators_0"], num_bins))
            ),
            dtype(
                (
                    "<f8",
                    (
                        num_blocking_levels,
                        num_blocking_levels,
                        L_max,
                        glue_params["num_operators_E"],
                        glue_params["num_operators_E"],
                        num_bins,
                    ),
                )
            ),
            dtype(
                (
                    "<f8",
                    (
                        num_blocking_levels,
                        num_blocking_levels,
                        L_max,
                        glue_params["num_operators_T"],
                        glue_params["num_operators_T"],
                        num_bins,
                    ),
                )
            ),
        )
        return (bctn, cor_L, vac_L, cor_L2, vac_L2, cor_0R, vac_0, cor_ER, cor_TR)
