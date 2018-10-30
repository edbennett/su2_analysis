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
    print(len(plaquettes))
    return plaquettes


def main():
    parser = ArgumentParser()

    parser.add_argument('--filename', required=True)
    parser.add_argument('--initial_configuration', default=0, type=int)
    parser.add_argument('--configuration_separation', default=1, type=int)
    parser.add_argument('--bootstrap_sample_count', default=200, type=int)
    parser.add_argument('--output_filename_prefix', default=None)
    parser.add_argument('--silent', action='store_true')
    args = parser.parse_args()

    if not args.output_filename_prefix:
        args.output_filename_prefix = args.filename + '_'

    plaquettes = get_plaquettes(args.filename)
    avr_plaquette, avr_plaquette_error = basic_bootstrap(
        plaquettes[args.initial_configuration::args.configuration_separation],
        bootstrap_sample_count=args.bootstrap_sample_count
    )

    write_results(
        filename=get_output_filename(
            args.output_filename_prefix, 'plaquette', filetype='dat'
        ),
        headers=('plaquette', 'plaquette_error'),
        values=((avr_plaquette, avr_plaquette_error),)
    )
    print(f"Plaquette: {avr_plaquette} ± {avr_plaquette_error}")


if __name__ == '__main__':
    main()
