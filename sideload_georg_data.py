from argparse import ArgumentParser
import pandas as pd
from autosu2.sideload import import_data

def do_import(filename):
    data = pd.read_csv(
        filename,
        delim_whitespace=True,
        comment='#',
        names=('Ensemble',
               'Ns', 'Nt', 'beta', 'am', 'start', 'end', 'total', 'end2',
               'g5_mass_value', 'g5_mass_uncertainty',
               'spin12_mass_value', 'spin12_mass_uncertainty')
    )
    import_data(data, observables=('spin12_mass',))


def main():
    parser = ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    do_import(args.filename)


if __name__ == '__main__':
    main()
