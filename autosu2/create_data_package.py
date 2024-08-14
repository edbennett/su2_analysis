#!/usr/bin/env python

from argparse import ArgumentParser
import logging

import h5py
from numpy import asarray, loadtxt, moveaxis, newaxis
import yaml

from glue_analysis.readers import read_correlators_fortran
from .avr_plaquette import get_plaquettes
from .data import get_correlators_from_raw, get_flows_from_raw
from .do_analysis import filter_complete, get_file_contents, get_subdirectory_name
from .glue import read_glue_correlation_matrices
from .modenumber import read_modenumber as read_modenumber_hirep
from .polyakov import get_loops_from_raw
from .provenance import get_basic_metadata, flatten_metadata
from .spin12 import get_correlators_spin12format


def plaq_getter(filename_base, group, **params):
    filename = filename_base + "out_hmc"
    plaquettes = get_plaquettes(filename)
    group.create_dataset("average_plaquette", data=plaquettes)


def gflow_getter(filename_base, group, **params):
    filename = filename_base + "out_wflow"
    subgroup = group.create_group("gradient_flow")
    trajectories, times, Eps, Ecs, Qs = get_flows_from_raw(filename)
    subgroup.create_dataset("trajectories", data=trajectories)
    subgroup.create_dataset("flow_time", data=times)
    subgroup.create_dataset("E_density_plaquette", data=Eps)
    subgroup.create_dataset("E_density_clover", data=Ecs)
    subgroup.create_dataset("topological_charge", data=Qs)


def polyakov_getter(filename_base, group, **params):
    filename = filename_base + "out_pl"
    subgroup = group.create_group("polyakov_loops")
    trajectories, plaquettes, polyakov_loops = get_loops_from_raw(filename, 0)
    subgroup.create_dataset("trajectories", data=trajectories)
    subgroup.create_dataset("directional_plaquettes", data=asarray(plaquettes))
    subgroup.create_dataset("loops", data=asarray(polyakov_loops))


def modenumber_getter_hirep(filename_base, group, **params):
    filename = filename_base + "out_modenumber"
    modenumbers = read_modenumber_hirep(filename)

    # nested dict with structure modenumbers[nu][trajectory] = nu
    # each modenumbers[nu] will have the same set of keys for trajectory
    # pick out an arbitrary one here
    upper_bounds = asarray(list(modenumbers.keys()))
    trajectories = asarray(list(next(iter(modenumbers.values())).keys()))

    # flatten the nested dict into a 2d array
    modenumbers_array = asarray(
        [
            [modenumbers[upper_bound][trajectory] for trajectory in trajectories]
            for upper_bound in upper_bounds
        ]
    )

    if len(modenumbers) == 0:
        return
    subgroup = group.create_group("modenumber")

    subgroup.create_dataset("upper_bounds", data=upper_bounds)
    subgroup.create_dataset("trajectories", data=trajectories)
    subgroup.create_dataset("modenumbers", data=modenumbers_array)


def modenumber_getter_colconf(filename_base, group, **params):
    filename = filename_base + "modenumber.dat"
    data = loadtxt(filename)
    upper_bounds = data[:, 0]
    modenumbers = data[:, 1:]

    subgroup = group.create_group("modenumber")

    # Trajectories are not indexed in colconf files
    subgroup.create_dataset("upper_bounds", data=upper_bounds)
    subgroup.create_dataset("modenumbers", data=modenumbers)


def modenumber_getter(filename_base, group, **params):
    if params.get("format", "hirep") == "hirep":
        modenumber_getter_hirep(filename_base, group, **params)
    elif params.get("format") == "colconf":
        modenumber_getter_colconf(filename_base, group, **params)
    else:
        logging.warning(f"Couldn't get mode numbers for {filename_base}")
        return


def mesons_getter(filename_base, group, **params):
    filename = filename_base + "out_corr"
    subgroup = group.create_group("correlators_hirep")
    correlators = get_correlators_from_raw(filename, group.attrs["T"])
    for valence_mass in set(correlators.valence_mass):
        mass_subgroup = subgroup.create_group(f"{valence_mass}")
        for channel in set(correlators.channel):
            subset_correlators = correlators[
                (correlators.valence_mass == valence_mass)
                & (correlators.channel == channel)
            ].drop(columns=["channel", "valence_mass"])
            if subset_correlators.empty:
                continue
            channel_subgroup = mass_subgroup.create_group(f"{channel}")
            channel_subgroup.create_dataset(
                "config_indices", data=subset_correlators.trajectory.to_numpy()
            )
            channel_subgroup.create_dataset(
                "correlators",
                data=subset_correlators.drop(columns=["trajectory"]).to_numpy(),
            )


def glueball_getter(filename_base, group, **params):
    if "num_bins" in params:
        fortran_glueball_getter(filename_base, group, **params)
    elif "cfg_count" in params:
        text_glueball_getter(filename_base, group, **params)
    else:
        logging.warning(f"Couldn't get glueballs for {filename_base}")


def fortran_glueball_getter(filename_base, group, **params):
    filename = filename_base + "glue_correlation_matrix"
    names = [
        "action",
        "cor_L",
        "vac_L",
        "cor_L2",
        "vac_L2",
        "cor_0R",
        "vac_0",
        "cor_ER",
        "cor_TR",
    ]
    default_num_momenta = 7

    correlation_matrices = read_glue_correlation_matrices(
        filename=filename,
        num_points_T=group.attrs["T"],
        num_configs=params.get("cfg_count", group.attrs["cfg_count"]),
        num_bins=params["num_bins"],
        num_momenta=default_num_momenta,
        num_blocking_levels=params["num_blocking_levels"],
    )

    subgroup = group.create_group("gluonic_correlation_matrices")
    for name, (matrix, num_operators) in zip(names, correlation_matrices):
        dataset = subgroup.create_dataset(name, data=matrix)
        dataset.attrs.update(
            {
                "cfg_count": params.get("cfg_count", group.attrs["cfg_count"]),
                "num_momenta": default_num_momenta,
                "num_bins": params["num_bins"],
                "num_blocking_levels": params["num_blocking_levels"],
                "num_operators": num_operators,
            }
        )


def text_glueball_getter(filename_base, group, **params):
    subgroup = group.create_group("gluonic_correlation_matrices")
    for channel, cor_suffix, vac_suffix in [
        ("A1++", "0R", "0"),
        ("torelon", "L", "L"),
        ("E++", "ER", "E"),
        ("T2++", "TR", "T"),
    ]:
        corr_filename = filename_base + "out_corr_" + channel
        vev_filename = filename_base + "out_vev_" + channel
        corr_ensemble = read_correlators_fortran(
            corr_filename,
            channel=channel,
            vev_filename=vev_filename,
            metadata={
                "NT": group.attrs["T"],
                "num_configs": params["cfg_count"],
            },
        )
        correlators = moveaxis(
            corr_ensemble.get_numpy(),
            [0, 1, 2, 3],
            [3, 0, 2, 1],
        )[newaxis, newaxis, ...]
        vevs = corr_ensemble.get_numpy_vevs().swapaxes(0, 1)[newaxis, ...]
        corr_dataset = subgroup.create_dataset(f"cor_{cor_suffix}", data=correlators)
        vev_dataset = subgroup.create_dataset(f"vac_{vac_suffix}", data=vevs)

        for dataset in corr_dataset, vev_dataset:
            dataset.attrs.update(
                {
                    "cfg_count": params["cfg_count"],
                    "num_momenta": 1,
                    "num_blocking_levels": 1,
                    "num_bins": corr_ensemble.num_samples,
                    "num_operators": corr_ensemble.num_internal,
                }
            )


def spin12_getter(filename_base, group, **params):
    filename = filename_base + "out_corr_spin12"
    file_contents = get_correlators_spin12format(filename)
    subgroup = group.create_group("correlators_spin12")
    for key, value in file_contents.items():
        if (
            key.startswith("polyakov_line")
            or key.endswith("plaquette")
            or key == "trajectories"
        ):
            subgroup.create_dataset(key, data=value)
        elif key in ("Meson_corr", "gluinoglue"):
            corr_subgroup = subgroup.create_group(key)
            metadata_subgroup = corr_subgroup.create_group("metadata")
            for meta_key, meta_value in value.pop("metadata").items():
                metadata_subgroup.create_dataset(meta_key, data=meta_value)
            for corr_name, corr_data in value.items():
                corr_subgroup.create_dataset(corr_name, data=corr_data)


data_getters = {
    "plaq": plaq_getter,
    "gflow": gflow_getter,
    "polyakov": polyakov_getter,
    "modenumber": modenumber_getter,
    "glueballs": glueball_getter,
    "mesons": mesons_getter,
    "spin12": spin12_getter,
}


def label_group(group, ensemble):
    keys = (
        "group_family",
        "group_rank",
        "representation",
        "Nf",
        "beta",
        "L",
        "T",
        "m",
        "first_cfg",
        "last_cfg",
        "cfg_count",
    )
    for key in keys:
        if key in ensemble:
            group.attrs[key] = ensemble[key]


def process_raw_to_hdf5(ensembles, metadata, filename):
    with h5py.File(filename, "w") as f:
        f.attrs.update(metadata)
        for label, ensemble in ensembles.items():
            print(label, "         ")
            group = f.create_group(label)
            label_group(group, ensemble)
            directory = get_subdirectory_name(ensemble)
            for name, getter in data_getters.items():
                measurement = ensemble.get(f"measure_{name}")
                if measurement:
                    print(name, "        ", end="\r")
                    filename_base = "raw_data/" + directory + "/"
                    if isinstance(measurement, dict):
                        getter(filename_base, group, **measurement)
                    else:
                        getter(filename_base, group)


def main():
    parser = ArgumentParser()
    parser.add_argument("ensembles_file")
    parser.add_argument("hdf5_file")
    args = parser.parse_args()

    ensembles = filter_complete(yaml.safe_load(get_file_contents(args.ensembles_file)))
    metadata = flatten_metadata(get_basic_metadata(args.ensembles_file))
    process_raw_to_hdf5(ensembles, metadata, args.hdf5_file)


if __name__ == "__main__":
    main()
