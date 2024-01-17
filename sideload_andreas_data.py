from argparse import ArgumentParser
import pandas as pd
from autosu2.sideload import import_data


def do_import(filename):
    data = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=1,
        names=(
            "Ensemble",
            "A1++_mass_value",
            "A1++_mass_uncertainty",
            "E++_mass_value",
            "E++_mass_uncertainty",
            "T2++_mass_value",
            "T2++_mass_uncertainty",
            "sqrtsigma_value",
            "sqrtsigma_uncertainty",
        ),
        na_values=("0.0000",),
    )

    import_data(data, observables=("A1++_mass", "E++_mass", "T2++_mass", "sqrtsigma"))


def main():
    parser = ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()

    do_import(args.filename)


if __name__ == "__main__":
    main()
