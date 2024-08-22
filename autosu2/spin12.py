#!/usr/bin/env python

import builtins
from collections import defaultdict
from glob import glob

import numpy as np


def tuple_to_complex(tuple_string):
    """
    Takes a string `tuple_string` of the form "(real_part,complex_part)" and
    returns a complex number.

    Examples:
    >>> tuple_to_complex('(-3.5,1.0)')
    (-3.5+1j)
    """
    components = tuple_string[1:-1].split(",")
    return complex(*map(float, components))


def process_corr_line(line, result, spares):
    channel = line[3]

    # Currently two parameters are fixed in the example output
    # Make sure these have not changed, as we assume as much in the following
    assert line[4:-5] == spares

    t = int(line[-4])
    if channel not in result:
        assert t == 0
        result[channel] = [float(line[-2])]
    else:
        assert len(result[channel]) >= t
        if len(result[channel]) == t:
            result[channel].append(float(line[-2]))
        else:
            result[channel][t] == float(line[-2])


def process_meson_corr_line(line, result):
    channel = line[3]
    if channel == "source":
        result["metadata"] = {
            f"source_{direction}": coordinate
            for direction, coordinate in zip("xyzt", map(int, line[5::2]))
        }
    elif channel == "itpropagator":
        result["metadata"][channel] = int(line[4])
    elif channel == "JN1":
        result["metadata"]["JN1"] = float(line[4])
        result["metadata"]["JN2"] = float(line[6])
    else:
        # Actual correlator
        process_corr_line(line, result, ["0", "0"])


def process_gluinoglue_line(line, result):
    channel = line[3]
    if channel == "total_inviter":
        result["metadata"] = {channel: int(line[4])}
    elif channel in ("source_t", "leveli", "levelj"):
        result["metadata"][channel] = int(line[4])
    elif channel.startswith("GGCorrTr"):
        # Actual correlator with spares
        process_corr_line(line, result, ["0", "0", ""])
    elif channel.startswith("WICorr"):
        process_corr_line(line, result, [])
    else:
        raise NotImplementedError(f"{channel} gluino-glue not yet handled")


def arrayify_corr(result):
    metadata = set(
        [key for cfg_result in result.values() for key in cfg_result["metadata"]]
    )
    new_result = {"metadata": {}}
    for attr in metadata:
        new_result["metadata"][attr] = np.asarray(
            [cfg_result["metadata"][attr] for _, cfg_result in sorted(result.items())]
        )

    channels = set([key for cfg_result in result.values() for key in cfg_result]) - {
        "metadata"
    }
    for channel in channels:
        new_result[channel] = np.asarray(
            [cfg_result[channel] for _, cfg_result in sorted(result.items())]
        )
    return new_result


def recursive_concatenate_dicts(to_concatenate):
    result = {}
    keys = set([key for single_dict in to_concatenate for key in single_dict])
    for key in keys:
        valid_subset = [
            single_dict[key] for single_dict in to_concatenate if key in single_dict
        ]
        types = set(map(type, valid_subset))
        if len(types) > 1:
            raise ValueError(f"Concatenation type mismatch: {types}")

        match types.pop():
            case np.ndarray:
                result[key] = np.concatenate(valid_subset)
            case builtins.dict:
                result[key] = recursive_concatenate_dicts(valid_subset)
            case invalid_type:
                raise ValueError(f"Don't know how to concatenate {invalid_type}")

    return result


def arrayify(all_results):
    all_new_results = []
    for results_idx, results in enumerate(all_results):
        new_results = {}
        for observable, observable_results in results.items():
            if observable.endswith("plaquette") or observable.startswith(
                "polyakov_line"
            ):
                new_results[observable] = np.asarray(
                    [value for _, value in sorted(observable_results.items())]
                )
            elif observable in ("Meson_corr", "gluinoglue"):
                new_results[observable] = arrayify_corr(observable_results)
            elif observable == "trajectories":
                new_results[observable] = np.vstack(
                    [
                        [results_idx] * len(observable_results),
                        observable_results,
                    ]
                ).T
            else:
                new_results[observable].append(observable_results)

        all_new_results.append(new_results)

    return recursive_concatenate_dicts(all_new_results)


def get_correlators_spin12format(filename_base):
    all_results = []

    for filename in glob(f"{filename_base}*"):
        cfg_indices = []
        results = defaultdict(lambda: defaultdict(dict))
        for i, line in enumerate(open(filename, "r")):
            split_line = line.split(":")
            if (not split_line) or (split_line[0] != "(MM)") or (len(split_line) < 3):
                continue
            cfg_index = int(split_line[1])
            cfg_indices.append(cfg_index)

            observable = split_line[2]
            if observable.endswith("plaquette"):
                results[observable][cfg_index] = float(split_line[3])
            elif observable.startswith("polyakov_line"):
                results[observable][cfg_index] = tuple_to_complex(split_line[3])
            elif observable == "Meson_corr":
                process_meson_corr_line(split_line, results[observable][cfg_index])
            elif observable == "gluinoglue":
                process_gluinoglue_line(split_line, results[observable][cfg_index])

        # Ensure that everything has a result
        keys_to_remove = []
        for observable_key, observable_results in results.items():
            # Some datasets are missing a large chunk of mesonic correlators
            # In order to maintain the data structure we have created elsewhere
            # ignore the mesonic correlators for these data
            if len(set(observable_results)) < 0.8 * len(set(cfg_indices)):
                keys_to_remove.append(observable_key)
            # Sometimes the last trajectory starts but does not complete
            else:
                for cfg_idx in set(cfg_indices):
                    if cfg_idx in observable_results:
                        continue
                    # Purge partial results
                    for _, other_observable_results in results.items():
                        if cfg_idx in other_observable_results:
                            del other_observable_results[cfg_idx]
                    cfg_indices = [idx for idx in cfg_indices if idx != cfg_idx]

        for key in keys_to_remove:
            del results[key]
        results["trajectories"] = np.asarray(sorted(set(cfg_indices)))

        all_results.append(results)

    return arrayify(all_results)
