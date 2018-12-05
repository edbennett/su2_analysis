from numpy import pi, asarray
from argparse import ArgumentParser

from bootstrap import basic_bootstrap
from data import write_results, get_output_filename


def get_plaquettes(filename):
    '''Given a HMC output filename, scan the file for all instances of
    plaquettes and return them in a list.'''

    plaquettes = []
    for line in open(filename, 'r'):
        if line.startswith("[MAIN][0]Plaquette: "):
            plaquettes.append(float(line.split(' ')[1]))
    return asarray(plaquettes)


def Z(plaquettes, beta, delta, bootstrap_sample_count=200):
    Z_values = 1 + 5 * delta * 8 / (beta * 4 * 16 * pi ** 2 * plaquettes)
    return basic_bootstrap(Z_values)


def main():
    parser = ArgumentParser()

    parser.add_argument('--filename', required=True)
    parser.add_argument('--beta', default=None, type=float)
    parser.add_argument('--initial_configuration', default=0, type=int)
    parser.add_argument('--configuration_separation', default=1, type=int)
    parser.add_argument('--bootstrap_sample_count', default=200, type=int)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--column_header_prefix', default='')
    args = parser.parse_args()
    if args.column_header_prefix != '':
        args.column_header_prefix = f'{args.column_header_prefix}_'

    if not args.output_filename_prefix:
        args.output_filename_prefix = args.filename + '_'

    plaquettes = get_plaquettes(args.filename)[
        args.initial_configuration::args.configuration_separation
    ]
    avr_plaquette, avr_plaquette_error = basic_bootstrap(
        plaquettes,
        bootstrap_sample_count=args.bootstrap_sample_count
    )

    headers = ['plaquette', 'plaquette_error']
    values_set = [(avr_plaquette, avr_plaquette_error)]

    if args.beta is not None:
        headers.extend(['Zv', 'Zv_error', 'Zav', 'Zav_error'])
        values_set.append(Z(
            plaquettes, -20.57, args.beta,
            bootstrap_sample_count=args.bootstrap_sample_count
        ))
        values_set.append(Z(
            plaquettes, -15.82, args.beta,
            bootstrap_sample_count=args.bootstrap_sample_count
        ))
    headers = [f'{args.column_header_prefix}{original_header}'
               for original_header in headers]

    write_results(
        filename=get_output_filename(
            args.output_filename_prefix, 'plaquette', filetype='dat'
        ),
        headers=headers,
        values_set=values_set,
    )
    if not args.silent:
        print(f"Plaquette: {avr_plaquette} ± {avr_plaquette_error}")
        if args.beta is not None:
            print(f"Z_v: {values_set[1][0]} ± {values_set[1][1]}")
            print(f"Z_v: {values_set[2][0]} ± {values_set[2][1]}")


if __name__ == '__main__':
    main()
