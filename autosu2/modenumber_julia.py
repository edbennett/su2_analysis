#!/usr/bin/env python3

from os.path import basename, dirname
from tempfile import NamedTemporaryFile

import io
import re
import subprocess

import pandas as pd

from .data import file_is_up_to_date
from .modenumber import read_modenumber


def reformat_modenumbers_hirep(filename):
    """Given the filename of a file containing modenumber data
    in HiRep format, read the data and return a file handle
    pointing to a pipe containing the data in a format readable
    by the Julia code."""

    modenumbers = read_modenumber(filename)
    reformatted_file = NamedTemporaryFile('w')
    for omega, omega_modenumbers in modenumbers.items():
        for nu in omega_modenumbers.values():
            print(f"[ {omega} ] = {nu}", file=reformatted_file)
    return reformatted_file


def read_modenumber_result(filename_or_file):
    result = {}

    if not isinstance(filename_or_file, io.IOBase):
        to_read = open(filename_or_file, "r")
        to_close = True
    else:
        to_read = filename_or_file
        to_close = False

    result["label"] = re.match(
        "# Results for ensemble (.*)",
        to_read.readline()
    ).groups()[0]
    result["gamma"], result["gamma_err"], result["syst_err"] = map(
        float,
        re.match(
            r"# γ\* = ([0-9.]+) ± ([0-9.]+)\s+\(syst\. err\. = ([0-9.]*)\)",
            to_read.readline(),
        ).groups()
    )
    result["raw_gammas"] = pd.read_csv(
        to_read,
        comment='#',
        delim_whitespace=True
    )

    if to_close:
        to_read.close()

    return result


def wrap_modenumber_fit_julia(
        modenumber_directory,
        ensemble,
        results_filename=None,
):
    modenumber_pars = ensemble["measure_modenumber"]
    volume = ensemble["L"] ** 3 * ensemble["T"]

    if (file_format := modenumber_pars["format"]) == "hirep":
        modenumber_filename = modenumber_directory + "/out_modenumber"
        formatted_modenumbers = reformat_modenumbers_hirep(modenumber_filename)
        formatted_modenumbers_filename = formatted_modenumbers.name
        modenumber_format = "HiRep"
    elif file_format == "colconf":
        modenumber_filename = modenumber_directory + "/modenumber.dat"
        formatted_modenumbers = None
        formatted_modenumbers_filename = modenumber_filename
        modenumber_format = "colconf"
    else:
        raise NotImplementedError(f"Format {file_format} is not recognised")

    if file_is_up_to_date(
            results_filename,
            compare_file=modenumber_filename
    ):
        return

    if results_filename:
        results_file = None
    else:
        results_file = NamedTemporaryFile('r')
        results_filename = results_file.name

    subprocess.run(
        [
            "julia",
            "--project=./modenumber.jl",
            "modenumber.jl/modenumber.jl",
            "--label",
            ensemble["descriptor"]["label"],
            "--beta",
            f"{ensemble['beta']}",
            "--mass",
            f"{ensemble['m']}",
            "--volume",
            f"{volume}",
            "--Npick",
            "80",
            "--omega_min_lower",
            f"{modenumber_pars['fit_omega_min']}",
            "--omega_min_step",
            "0.01",
            "--omega_min_upper",
            f"{modenumber_pars['fit_omega_max']}",
            "--delta_omega_lower",
            f"{modenumber_pars['fit_window_length_min']}",
            "--delta_omega_step",
            "0.01",
            "--delta_omega_upper",
            f"{modenumber_pars['fit_window_length_max']}",
            "--output",
            results_filename,
            "--input_path",
            dirname(formatted_modenumbers_filename),
            "--input_filename",
            basename(formatted_modenumbers_filename),
            "--input_type",
            modenumber_format,
        ]
    ).check_returncode()

    if results_file:
        results = read_modenumber_result(results_file)
        results_file.close()
    else:
        results = read_modenumber_result(results_filename)

    if formatted_modenumbers:
        formatted_modenumbers.close()

    return results


if __name__ == "__main__":
    raise NotImplementedError("No command line interface implemented here")
