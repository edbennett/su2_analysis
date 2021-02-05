from argparse import ArgumentParser
import pandas as pd
from numpy import isnan
from autosu2.sideload import import_data

def do_import(filename):
    data = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=1,
        names=('Ensemble',
               'App_mass_value', 'App_mass_uncertainty',
               'Epp_mass_value', 'Epp_mass_uncertainty',
               'Tpp_mass_value', 'Tpp_mass_uncertainty',
               'sqrtsigma_value', 'sqrtsigma_uncertainty'),
        na_values=('0.0000',)
    )

    import_data(
        data,
        observables=('App_mass', 'Epp_mass', 'Tpp_mass', 'sqrtsigma')
    )


def main():
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    do_import(args.filename)


if __name__ == '__main__':
    main()
