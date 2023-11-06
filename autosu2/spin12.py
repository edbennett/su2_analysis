#!/usr/bin/env python

from collections import defaultdict

from numpy import asarray


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
        new_result["metadata"][attr] = asarray(
            [cfg_result["metadata"][attr] for _, cfg_result in sorted(result.items())]
        )

    channels = set([key for cfg_result in result.values() for key in cfg_result]) - {
        "metadata"
    }
    for channel in channels:
        new_result[channel] = asarray(
            [cfg_result[channel] for _, cfg_result in sorted(result.items())]
        )
    return new_result


def arrayify(results):
    new_results = {}
    for observable, observable_results in results.items():
        if observable.endswith("plaquette") or observable.startswith("polyakov_line"):
            new_results[observable] = asarray(
                [value for _, value in sorted(observable_results.items())]
            )
        elif observable in ("Meson_corr", "gluinoglue"):
            new_results[observable] = arrayify_corr(observable_results)
        else:
            new_results[observable] = observable_results
    return new_results


def get_correlators_spin12format(filename):
    results = defaultdict(lambda: defaultdict(dict))

    cfg_indices = set()

    for i, line in enumerate(open(filename, "r")):
        split_line = line.split(":")
        if (not split_line) or split_line[0] != "(MM)":
            continue
        cfg_index = int(split_line[1])
        cfg_indices.add(cfg_index)

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
        if len(observable_results) < 0.8 * len(cfg_indices):
            keys_to_remove.append(observable_key)
        else:
            assert set(observable_results.keys()) == cfg_indices

    for key in keys_to_remove:
        del results[key]
    results["trajectories"] = asarray(sorted(cfg_indices))
    return arrayify(results)
