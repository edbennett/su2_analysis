import yaml

from argparse import ArgumentParser
from os import listdir, makedirs
from importlib import import_module
from datetime import datetime
from os.path import getmtime
import logging

from .data import get_subdirectory_name
from .db import is_complete_descriptor, describe_ensemble, get_dataframe

from .w0 import plot_measure_and_save_w0, DEFAULT_W0
from .t0 import plot_measure_and_save_sqrt_8t0, DEFAULT_E0
from .Q import plot_measure_and_save_Q
from .avr_plaquette import measure_and_save_avr_plaquette
from .fit_correlation_function import plot_measure_and_save_mesons, Incomplete
from .fit_spin12 import plot_measure_and_save_spin12
from .fit_effective_mass import plot_measure_and_save_mpcac
from .fit_glue import plot_measure_and_save_glueballs, select_2plusplus_state
from .polyakov import fit_plot_and_save_polyakov_loops
from .provenance import stamp_provenance
from .modenumber import do_modenumber_fit
from .modenumber_julia import wrap_modenumber_fit_julia
from .sideload import callback_string_tension, import_data_sql, import_data_csv
from .modenumber_aic import do_modenumber_fit_aic


DEBUG = True


def filter_complete(ensembles):
    return {
        label: ensemble
        for label, ensemble in ensembles.items()
        if is_complete_descriptor(ensemble)
    }


def get_file_contents(filename):
    with open(filename, "r") as f:
        return f.read()


def do_single_analysis(
    label, ensemble, ensembles_date=datetime.now, skip_mesons=False, **kwargs
):
    ensemble["descriptor"] = describe_ensemble(ensemble, label)
    subdirectory = get_subdirectory_name(ensemble)
    makedirs("processed_data/" + subdirectory, exist_ok=True)

    if DEBUG:
        print("Processing", subdirectory)

    if ensemble.get("measure_gflow", False):
        # Gradient flow: Q
        if DEBUG:
            print("  - Q")
        result = plot_measure_and_save_Q(
            simulation_descriptor=ensemble["descriptor"],
            flows_file=f"raw_data/{subdirectory}/out_wflow",
            output_file_history=f"processed_data/{subdirectory}/Q.pdf",
            output_file_autocorr=f"processed_data/{subdirectory}/Q_corr.pdf",
            reader=ensemble["measure_gflow"],
        )
        if result and DEBUG:
            print("    {}".format(" ".join([f"{k}: {v}" for k, v in result.items()])))
        else:
            print("    Already up to date")

        # Gradient flow: w0
        if DEBUG:
            print("  - w0")
        result = plot_measure_and_save_w0(
            W0=DEFAULT_W0,
            simulation_descriptor=ensemble["descriptor"],
            filename=f"raw_data/{subdirectory}/out_wflow",
            plot_filename=(f"processed_data/{subdirectory}/flows.pdf"),
            reader=ensemble["measure_gflow"],
        )
        if result and DEBUG:
            w0p, w0c = result
            print("    w0p:", w0p, "w0c:", w0c)
        elif DEBUG:
            print("    Already up to date")

        if ensemble["beta"] == 2.1:
            # Generate extra W0 values for figure 1
            # Could be made more efficient by splitting
            # `plot_measure_and_save_w0` into two functions
            for W0 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                result = plot_measure_and_save_w0(
                    W0=W0,
                    simulation_descriptor=ensemble["descriptor"],
                    filename=f"raw_data/{subdirectory}/out_wflow",
                    reader=ensemble["measure_gflow"],
                )
                if result and DEBUG:
                    w0p, w0c = result
                    print("    W0:", W0, "| w0p:", w0p, "w0c:", w0c)
                elif DEBUG:
                    print("    W0:", W0, "also already up to date")

        # Gradient flow: t0
        if DEBUG:
            print("  - t0")
        result = plot_measure_and_save_sqrt_8t0(
            E0=DEFAULT_E0,
            simulation_descriptor=ensemble["descriptor"],
            filename=f"raw_data/{subdirectory}/out_wflow",
            plot_filename=(f"processed_data/{subdirectory}/flows_t0.pdf"),
            reader=ensemble["measure_gflow"],
        )
        if result and DEBUG:
            s8t0p, s8t0c = result
            print("    sqrt(8t0p):", s8t0p, "sqrt(8t0c):", s8t0c)
        elif DEBUG:
            print("    Already up to date")

        if ensemble["beta"] == 2.1:
            # Generate extra E0 values for figure 1
            # Could be made more efficient by splitting
            # `plot_measure_and_save_w0` into two functions
            for E0 in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
                result = plot_measure_and_save_sqrt_8t0(
                    E0=E0,
                    simulation_descriptor=ensemble["descriptor"],
                    filename=f"raw_data/{subdirectory}/out_wflow",
                    reader=ensemble["measure_gflow"],
                )
                if result and DEBUG:
                    s8t0p, s8t0c = result
                    print("    E0:", E0, "| sqrt(8t0p):", s8t0p, "sqrt(8t0c):", s8t0c)
                elif DEBUG:
                    print("    E0:", E0, "also already up to date")

    if ensemble.get("measure_plaq", False):
        # HMC history: plaquette
        if DEBUG:
            print("  - Plaquette")
        result = measure_and_save_avr_plaquette(
            simulation_descriptor=ensemble["descriptor"],
            filename=f"raw_data/{subdirectory}/out_hmc",
        )
        if result and DEBUG:
            plaq, Zv, Zav = result
            print("    plaq:", plaq, "Zv:", Zv, "Zav:", Zav)
        elif DEBUG:
            print("    Already up to date")

    if isinstance(ensemble.get("measure_mesons"), dict) and not skip_mesons:
        # Mesonic observables
        for channel_name, channel_parameters in ensemble["measure_mesons"].items():
            if DEBUG:
                print(f"  - Mesons, {channel_name}")
            try:
                result = plot_measure_and_save_mesons(
                    simulation_descriptor=ensemble["descriptor"],
                    correlator_filename=f"raw_data/{subdirectory}/out_corr",
                    channel_name=channel_name,
                    meson_parameters=channel_parameters,
                    parameter_date=ensembles_date,
                    output_filename_prefix=f"processed_data/{subdirectory}/",
                )
            except Incomplete as ex:
                print(f"    INCOMPLETE: {ex.message}")
            else:
                if result and DEBUG:
                    print("   ", result)
                else:
                    print("    Already up to date")

    if isinstance(spin12_params := ensemble.get("measure_spin12"), dict):
        # Spin-1/2 state
        if DEBUG:
            print("  - Spin-1/2")
        try:
            result = plot_measure_and_save_spin12(
                simulation_descriptor=ensemble["descriptor"],
                correlator_directory=f"raw_data/{subdirectory}",
                spin12_parameters=spin12_params,
                parameter_date=ensembles_date,
                output_filename_prefix=f"processed_data/{subdirectory}/",
            )
        except Incomplete as ex:
            print(f"    INCOMPLETE: {ex.message}")
        else:
            if result and DEBUG:
                print("   ", result)
            else:
                print("    Already up to date")

    if isinstance(ensemble.get("measure_glueballs"), dict):
        # Glueballs
        glue_channels = ["torelon", "A1++", "E++", "T2++"]
        for channel_name, channel_parameters in ensemble["measure_glueballs"].items():
            if channel_name not in glue_channels:
                continue
            if DEBUG:
                print(f"  - Glueballs, {channel_name}")
            try:
                result = plot_measure_and_save_glueballs(
                    simulation_descriptor=ensemble["descriptor"],
                    correlator_filename=f"raw_data/{subdirectory}/out_corr_{channel_name}",
                    vev_filename=f"raw_data/{subdirectory}/out_vev_{channel_name}",
                    num_configs=ensemble["measure_glueballs"].get("cfg_count")
                    or ensemble["cfg_count"],
                    channel_name=channel_name,
                    glue_parameters=channel_parameters,
                    parameter_date=ensembles_date,
                    output_filename_prefix=f"processed_data/{subdirectory}/",
                )
            except Incomplete as ex:
                print(f"    INCOMPLETE: {ex.message}")
            else:
                if DEBUG:
                    if result:
                        print("   ", result)
                    else:
                        print("    Already up to date")

        result = select_2plusplus_state(
            simulation_descriptor=ensemble["descriptor"],
            Epp_params=ensemble["measure_glueballs"].get("E++", {}),
            T2pp_params=ensemble["measure_glueballs"].get("T2++", {}),
        )
        if DEBUG:
            print(f"  - Glueballs, selecting 2++")
            if result:
                print(f"    {result:.02uS}")
            else:
                print("    no data")

    if ensemble.get("measure_pcac", False) and not skip_mesons:
        # Mesonic observables
        if DEBUG:
            print("  - PCAC mass")
        try:
            result = plot_measure_and_save_mpcac(
                simulation_descriptor=ensemble["descriptor"],
                correlator_filename=f"raw_data/{subdirectory}/out_corr",
                meson_parameters=ensemble["measure_pcac"],
                parameter_date=ensembles_date,
                output_filename_prefix=f"processed_data/{subdirectory}/",
            )
        except Incomplete as ex:
            print(f"    INCOMPLETE: {ex.message}")
        else:
            if result and DEBUG:
                print("   ", result)
            else:
                print("    Already up to date")

    if ensemble.get("measure_polyakov", False):
        # Simple Polyakov loop analysis for centre symmetry
        if DEBUG:
            print("  - Polyakov loops")
        fit_results = fit_plot_and_save_polyakov_loops(
            simulation_descriptor=ensemble["descriptor"],
            filename=f"raw_data/{subdirectory}/out_pl",
            plot_filename=(f"processed_data/{subdirectory}/polyakov.pdf"),
            do_fit=False,
        )
        if DEBUG and fit_results:
            for direction, result in enumerate(fit_results):
                print(f"    Direction {direction}:", result)

    measure_modenumber = ensemble.get("measure_modenumber", None)
    # Mode number analysis for anomalous dimension
    if measure_modenumber and measure_modenumber["method"] == "julia":
        if DEBUG:
            print("  - Mode number (Julia)")

        modenumber_result = wrap_modenumber_fit_julia(
            ensemble=ensemble,
            modenumber_directory=f"raw_data/{subdirectory}",
            results_filename=f"processed_data/{subdirectory}/modenumber_fit_julia.csv",
        )
        if (modenumber_result is None) and DEBUG:
            print("    Already up to date")

    elif measure_modenumber and measure_modenumber["method"] == "aic":
        result = do_modenumber_fit_aic(
            ensemble=ensemble,
            filename=f"raw_data/{subdirectory}/out_modenumber",
            # boot_gamma=read_modenumber_result(f'processed_data/{subdirectory}/modenumber_fit_julia.csv')['gamma'],
            boot_gamma=0.5,
            plot_directory=f"processed_data/{subdirectory}",
        )
        if DEBUG:
            print("  - gamma*")
        if result and DEBUG:
            print("   ", result)
        else:
            print("    Already up to date")

    elif measure_modenumber:
        if DEBUG:
            print("  - Mode number")
        modenumber_result = do_modenumber_fit(
            f"raw_data/{subdirectory}/out_modenumber",
            f"processed_data/{subdirectory}/modenumber_fit.csv",
        )
        if (modenumber_result is None) and DEBUG:
            print("    Already up to date")


def do_analysis(ensembles, single_ensemble=None, **kwargs):
    for label, ensemble in ensembles.items():
        if (not single_ensemble) or label == single_ensemble:
            do_single_analysis(label, ensemble, **kwargs)


def output_results(only=None, ensembles=None):
    data = get_dataframe()

    for object_type in ("table", "plot", "data"):
        objects = [
            import_module(f"autosu2.{object_type}_specs." + module[:-3])
            for module in listdir(f"autosu2/{object_type}_specs")
            if module[-3:] == ".py" and module[0] != "."
        ]

        for object in objects:
            if "__" in object.__name__:
                continue
            if only and only not in object.__name__:
                continue
            try:
                print(f"Generating for {object.__name__}")
                object.generate(data, ensembles=ensembles)
            except AttributeError as ex:
                if "has no attribute 'generate'" in str(ex):
                    print(f"Module {object.__name__} has no generate function.")
                else:
                    raise ex


def main():
    parser = ArgumentParser()
    parser.add_argument("--ensembles", default="ensembles.yaml")
    parser.add_argument("--skip_mesons", action="store_true")
    parser.add_argument("--skip_calculation", action="store_true")
    parser.add_argument("--skip_output", action="store_true")
    parser.add_argument("--only", default=None)
    parser.add_argument("--sideload_csv", default=None)
    parser.add_argument("--sideload_sql", default=None)
    parser.add_argument("--quenched", action="store_true")
    parser.add_argument("--single_ensemble", default=None)
    args = parser.parse_args()

    if DEBUG:
        logging.getLogger().setLevel(logging.INFO)

    ensembles = filter_complete(yaml.safe_load(get_file_contents(args.ensembles)))
    ensembles_date = datetime.fromtimestamp(getmtime(args.ensembles))

    if args.skip_calculation or args.only:
        print("Skipping calculation as requested")
    else:
        do_analysis(
            ensembles,
            ensembles_date=ensembles_date,
            skip_mesons=args.skip_mesons,
            only=args.only,
            quenched=args.quenched,
            single_ensemble=args.single_ensemble,
        )

    if args.sideload_csv:
        import_data_csv(args.sideload_csv)

    if args.sideload_sql:
        import_data_sql(
            args.sideload_sql,
            list(ensembles.keys()),
            ["App_mass", "Epp_mass", "Tpp_mass", "sqrtsigma", "spin12_mass"],
            observable_names={
                "App_mass": "A1++_mass",
                "Epp_mass": "E++_mass",
                "Tpp_mass": "T2++_mass",
            },
            callback=callback_string_tension,
        )

    if not args.skip_output:
        print("Outputting results:")
        output_results(args.only, ensembles=ensembles)

    if not (
        args.only
        or args.skip_mesons
        or args.skip_output
        or args.skip_calculation
        or args.single_ensemble
    ):
        stamp_provenance(ensembles_filename=args.ensembles)


if __name__ == "__main__":
    main()
