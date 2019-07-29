from numpy import pi, asarray
from argparse import ArgumentParser

from .bootstrap import basic_bootstrap
# from .data import write_results, get_output_filename
from .db import measurement_is_up_to_date, add_measurement
from .data import get_filename


def get_plaquettes(filename):
    '''Given a HMC output filename, scan the file for all instances of
    plaquettes and return them in a list.'''

    plaquettes = []
    for line in open(filename, 'r'):
        if line.startswith("[MAIN][0]Plaquette: "):
            plaquettes.append(float(line.split(' ')[1]))
    return asarray(plaquettes)


def Z(plaquettes, beta, delta):
    Z_values = 1 + 5 * delta * 8 / (beta * 4 * 16 * pi ** 2 * plaquettes)
    return basic_bootstrap(Z_values)


def measure_and_save_avr_plaquette(simulation_descriptor=None,
                                   initial_configuration=50,
                                   configuration_separation=3,
                                   filename=None,
                                   filename_formatter=None,
                                   force=False):
    filename = get_filename(simulation_descriptor,
                            filename_formatter, filename)

    if (simulation_descriptor
        and not force
        and (measurement_is_up_to_date(simulation_descriptor, 'avr_plaquette',
                                       compare_file=filename)
             and measurement_is_up_to_date(simulation_descriptor, 'Zv',
                                           compare_file=filename)
             and measurement_is_up_to_date(simulation_descriptor, 'Zav',
                                           compare_file=filename))):
        # Already up to date
        return

    plaquettes = get_plaquettes(filename)[
        initial_configuration::configuration_separation
    ]
    avr_plaquette, avr_plaquette_error = basic_bootstrap(
        plaquettes
    )
    results = [(avr_plaquette, avr_plaquette_error)]

    if simulation_descriptor:
        results.append(Z(plaquettes, -20.57, simulation_descriptor['beta']))
        results.append(Z(plaquettes, -15.82, simulation_descriptor['beta']))

        quantities = ('avr_plaquette', 'Zv', 'Zav')
        for quantity, result in zip(quantities, results):
            add_measurement(simulation_descriptor, quantity, *result)

    return results


def main():
    parser = ArgumentParser()

    parser.add_argument('--filename', required=True)
    parser.add_argument('--initial_configuration', default=0, type=int)
    parser.add_argument('--configuration_separation', default=1, type=int)
    parser.add_argument('--silent', action='store_true')
#    parser.add_argument('--column_header_prefix', default='')
    args = parser.parse_args()
#    if args.column_header_prefix != '':
#        args.column_header_prefix = f'{args.column_header_prefix}_'

#    if not args.output_filename_prefix:
#        args.output_filename_prefix = args.filename + '_'

    avr_plaquette = measure_and_save_avr_plaquette(
        filename=args.filename,
        initial_configuration=args.initial_configuration,
        configuration_separation=args.configuration_separation
    )

    if not args.silent and avr_plaquette:
        print(f"Plaquette: {avr_plaquette[0]} ± {avr_plaquette[1]}")


if __name__ == '__main__':
    main()
