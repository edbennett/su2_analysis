from re import compile

from numpy import pi, asarray
from argparse import ArgumentParser

from .bootstrap import basic_bootstrap

# from .data import write_results, get_output_filename
from .db import measurement_is_up_to_date, add_measurement
from .data import get_filename

C_F = 5 / 4
DELTA_SIGMA_ONE = -12.82
DELTA_G5GMU = -3.0
DELTA_GMU = -7.75

TRAJECTORY_GETTER = compile(r"\[MAIN\]\[0\]Trajectory #(?P<trajectory>[0-9]+)")


def get_plaquettes(filename, first_trajectory=0):
    """Given a HMC output filename, scan the file for all instances of
    plaquettes and return them in a list."""

    plaquettes = []
    current_trajectory = 0
    for line in open(filename, "r"):
        trajectory = TRAJECTORY_GETTER.match(line)
        if trajectory:
            current_trajectory = int(trajectory.group("trajectory"))
        if current_trajectory > first_trajectory and line.startswith(
            "[MAIN][0]Plaquette: "
        ):
            plaquettes.append(float(line.split(" ")[1]))
    return asarray(plaquettes)


def Z(plaquettes, delta, beta):
    #  Z     = 1 + C_F  \Delta          \tilde{g}^2      /   {16 \pi^2}
    Z_values = 1 + C_F * delta * (8 / beta / plaquettes) / (16 * pi**2)
    return basic_bootstrap(Z_values)


def measure_and_save_avr_plaquette(
    simulation_descriptor=None,
    initial_configuration=50,
    configuration_separation=3,
    filename=None,
    filename_formatter=None,
    force=False,
):
    filename = get_filename(simulation_descriptor, filename_formatter, filename)

    if (
        simulation_descriptor
        and not force
        and (
            measurement_is_up_to_date(
                simulation_descriptor, "avr_plaquette", compare_file=filename
            )
            and measurement_is_up_to_date(
                simulation_descriptor, "Zv", compare_file=filename
            )
            and measurement_is_up_to_date(
                simulation_descriptor, "Zav", compare_file=filename
            )
        )
    ):
        # Already up to date
        return

    if simulation_descriptor:
        initial_configuration = simulation_descriptor.get(
            "first_cfg", initial_configuration
        )

    plaquettes = get_plaquettes(filename, initial_configuration)

    avr_plaquette, avr_plaquette_error = basic_bootstrap(plaquettes)
    results = [(avr_plaquette, avr_plaquette_error)]

    if simulation_descriptor:
        results.append(
            Z(plaquettes, DELTA_SIGMA_ONE + DELTA_GMU, simulation_descriptor["beta"])
        )
        results.append(
            Z(plaquettes, DELTA_SIGMA_ONE + DELTA_G5GMU, simulation_descriptor["beta"])
        )

        quantities = ("avr_plaquette", "Zv", "Zav")
        for quantity, result in zip(quantities, results):
            add_measurement(simulation_descriptor, quantity, *result)

    return results


def main():
    parser = ArgumentParser()

    parser.add_argument("--filename", required=True)
    parser.add_argument("--initial_configuration", default=0, type=int)
    parser.add_argument("--configuration_separation", default=1, type=int)
    parser.add_argument("--silent", action="store_true")
    #    parser.add_argument('--column_header_prefix', default='')
    args = parser.parse_args()
    #    if args.column_header_prefix != '':
    #        args.column_header_prefix = f'{args.column_header_prefix}_'

    #    if not args.output_filename_prefix:
    #        args.output_filename_prefix = args.filename + '_'

    avr_plaquette = measure_and_save_avr_plaquette(
        filename=args.filename,
        initial_configuration=args.initial_configuration,
        configuration_separation=args.configuration_separation,
    )

    if not args.silent and avr_plaquette:
        print(f"Plaquette: {avr_plaquette[0]} ± {avr_plaquette[1]}")


if __name__ == "__main__":
    main()
