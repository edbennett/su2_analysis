from datetime import datetime
from os.path import getmtime
from csv import writer, QUOTE_MINIMAL
from re import findall, match
from functools import lru_cache
from numbers import Number

from pandas import read_csv, DataFrame, concat
from numpy import asarray, float64


def get_subdirectory_name(descriptor):
    return ("nf{Nf}{rep_suffix}/b{beta}{m_suffix}/{T}x{L}{directory_suffix}").format(
        Nf=descriptor["Nf"],
        L=descriptor["L"],
        T=descriptor["T"],
        beta=descriptor["beta"],
        m_suffix=f'/m{descriptor["m"]}' if "m" in descriptor else "",
        rep_suffix=(
            f'_{descriptor["representation"]}' if "representation" in descriptor else ""
        ),
        directory_suffix=(
            f'_{descriptor["directory_suffix"]}'
            if "directory_suffix" in descriptor
            else ""
        ),
    )


def write_results(
    filename, headers, values_set, channel_name="", many=False, extras_set=None
):
    if channel_name:
        channel_name = f"{channel_name}_"
    if not many:
        values_set = (values_set,)
    if not extras_set:
        extras_set = [[[]]] * len(values_set)

    with open(filename, "w", newline="") as csvfile:
        csv_writer = writer(
            csvfile, delimiter="\t", quoting=QUOTE_MINIMAL, lineterminator="\n"
        )
        csv_writer.writerow((f"{channel_name}{header}" for header in headers))
        #        import pdb; pdb.set_trace()
        for values, extras in zip(values_set, *zip(*extras_set)):
            csv_writer.writerow(
                [value for value_pair in values for value in value_pair] + extras
            )


def read_db(filename):
    return read_csv(filename, sep="\s+", na_values="?")


def get_filename(simulation_descriptor, filename_formatter, filename, optional=False):
    """Check that the correct arguments have been given, and return the
    filename to use."""

    if filename_formatter and filename:
        raise ValueError(
            "A filename formatter and a filename cannot both be specified."
        )
    if not filename_formatter and not filename and not optional:
        raise ValueError(
            "Either a filename formatter or a filename  must be specified."
        )
    if filename_formatter:
        if not simulation_descriptor:
            raise ValueError(
                "If a filename formatter is specified, then a simulation"
                "descriptor must also be specified."
            )
        filename = filename_formatter.format(**simulation_descriptor)

    return filename


def get_output_filename(basename, type, channel="", tstart="", tend="", filetype="pdf"):
    if channel:
        channel = f"_{channel}"
    if tstart:
        tstart = f"_{tstart}"
    if tend:
        tend = f"_{tend}"
    if tstart and not tend:
        tend = "_XX"
    if tend and not tstart:
        tstart = "_XX"

    return f"{basename}{type}{channel}{tstart}{tend}.{filetype}"


def bin_correlator(correlator, bin_size):
    binned_correlator = DataFrame()
    num_bins = len(correlator) // bin_size
    num_configs = num_bins * bin_size
    for column_label, column_data in correlator.iteritems():
        binned_correlator[column_label] = (
            column_data.values[:num_configs].reshape((num_bins, bin_size)).mean(axis=1)
        )
    return binned_correlator


def get_single_raw_correlator_set(
    all_correlators,
    channels,
    NT,
    NS,
    valence_masses,
    sign=+1,
    initial_configuration=0,
    bin_size=1,
):
    correlator_set = []

    all_relevant_correlators = concat(
        (
            all_correlators[
                (all_correlators.channel == channel)
                & (all_correlators.trajectory >= initial_configuration)
            ]
            for channel in channels
        )
    )
    all_relevant_correlators.drop(["channel"], axis=1, inplace=True)

    for valence_mass in valence_masses:
        target_correlator = all_relevant_correlators[
            all_relevant_correlators.valence_mass == valence_mass
        ]

        target_correlator.drop(["trajectory", "valence_mass"], axis=1, inplace=True)
        target_correlator[NT] = target_correlator[0]

        for column in range(NT // 2 + 1):
            target_correlator[column] += sign * target_correlator[NT - column]
            target_correlator[column] /= 2
        target_correlator.drop(range(NT // 2 + 1, NT + 1), axis=1, inplace=True)

        # Fits require positive data; flip sign if necessary
        # The first point is frequently a different sign and orders of
        # magnitude larger, so ignore it.
        if target_correlator[range(1, NT // 2)].mean().mean() < 0:
            target_correlator = -target_correlator

        correlator_set.append(bin_correlator(target_correlator, bin_size) * NS**3)

    return correlator_set


@lru_cache(maxsize=8)
def get_correlators_from_raw(filename, NT):
    configuration_index = 0
    reported_configuration_index = ""
    data = []
    with open(filename) as f:
        for line in f:
            if not line.startswith("[MAIN][0]conf"):
                continue
            line_content = line.split()
            if line_content[3] == "SINGLET":
                # We can't deal with these yet
                continue
            if len(line_content) == NT + 5:
                # We mix multiple versions of the code; some include a
                # tag indicating the algorithm used and others don't.
                # We don't have any old output files that don't use the
                # DEFAULT_SEMWALL, so we detect too-short lines and add this.
                # In principle we could also delete it if present, but this
                # is hopefully more extensible if we ever need to make
                # decisions based on this tag
                line_content.insert(3, "DEFAULT_SEMWALL")
            if line_content[1] != reported_configuration_index:
                configuration_index += 1
                reported_configuration_index = line_content[1]
            valence_mass = float64(line_content[2].split("=")[1])
            channel_name = line_content[5].replace("=", "")

            assert len(line_content[6:]) == NT

            data.append(
                (
                    configuration_index,
                    valence_mass,
                    channel_name,
                    *float64(line_content[6:]),
                )
            )
    return DataFrame.from_records(
        data, columns=("trajectory", "valence_mass", "channel") + tuple(range(NT))
    )


def get_correlators_from_filtered(filename, NT):
    with open(filename) as f:
        number_of_columns = len(f.readline().split())

    assert NT == number_of_columns - 3

    column_names = ["trajectory", "valence_mass", "channel"] + list(range(NT))

    all_correlators = read_csv(filename, names=column_names, sep="\s+")

    return all_correlators


def get_target_correlator(
    filename,
    channel_sets,
    NT,
    NS,
    signs,
    initial_configuration=0,
    bin_size=1,
    from_raw=True,
):
    if from_raw:
        get_file_data = get_correlators_from_raw
    else:
        get_file_data = get_correlators_from_filtered

    all_correlators = get_file_data(filename, NT)
    valence_masses = sorted(set(all_correlators.valence_mass))
    all_channels = set(all_correlators.channel)

    configuration_count = (
        len(all_correlators) // len(all_channels) // len(valence_masses) // bin_size - 1
    )

    used_configuration_count = configuration_count - initial_configuration + 1

    target_correlator_sets = []

    for channels, sign in zip(channel_sets, signs):
        target_correlator_sets.append(
            get_single_raw_correlator_set(
                all_correlators,
                channels,
                NT,
                NS,
                valence_masses,
                sign,
                initial_configuration,
                bin_size,
            )
        )
        for raw_correlator in target_correlator_sets[-1]:
            assert (
                len(raw_correlator) - len(channels)
                <= used_configuration_count * len(channels)
                <= len(raw_correlator)
            )
        assert len(target_correlator_sets[-1]) == len(valence_masses)

    return map(list, zip(*target_correlator_sets)), valence_masses


def get_flows(filename):
    data = read_csv(filename, sep="\s+", names=["n", "t", "Ep", "Ec", "Q"])
    times = asarray(sorted(set(data.t)))
    Eps = asarray([data[data.n == n].Ep.values for n in set(data.n)])
    Ecs = asarray([data[data.n == n].Ec.values for n in set(data.n)])

    return times, Eps, Ecs


def bin_flows(flows, bin_size):
    flows = asarray(flows)
    num_bins = flows.shape[0] // bin_size
    return (
        flows[: num_bins * bin_size, :]
        .reshape((num_bins, bin_size, flows.shape[1]))
        .mean(axis=1)
    )


def complete_trajectory(
    trajectories, Eps, Ecs, Qs, trajectory, current_Eps, current_Ecs, current_Q, times
):
    if len(current_Eps) == len(current_Ecs) == len(times):
        if (not isinstance(current_Q, list)) or (len(Qs) == len(times)):
            trajectories.append(trajectory)
            Eps.append(current_Eps)
            Ecs.append(current_Ecs)
            Qs.append(current_Q)


@lru_cache(maxsize=8)
def get_flows_from_raw(filename, bin_size=1, limit_t_for_Q=None, raw_Qs=False):
    trajectories = []
    Eps = []
    Ecs = []
    current_Eps = None
    current_Ecs = None
    times = []
    Qs = []
    current_Q = None
    times_acquired = None
    if isinstance(limit_t_for_Q, Number):
        t_max_for_Q = limit_t_for_Q
    else:
        t_max_for_Q = None

    with open(filename) as f:
        for line in f.readlines():
            line_contents = line.split()
            if (
                line_contents[0] == "[IO][0]Configuration"
                and line_contents[2] == "read"
            ):
                trajectory = int(findall(r".*n(\d+)]", line_contents[1])[0])
                continue

            if line_contents[0] == "[GEOMETRY][0]Global":
                NT, NX, NY, NZ = map(
                    int,
                    match(
                        "([0-9]+)x([0-9]+)x([0-9]+)x([0-9]+)", line_contents[3]
                    ).groups(),
                )
                NL = min(NX, NY, NZ)
                if limit_t_for_Q == "L/2":
                    t_max_for_Q = (NL / 2) ** 2 / 8

            if line_contents[0] != "[WILSONFLOW][0]WF":
                continue
            if line_contents[1].startswith("(ncnfg"):
                del line_contents[3]
            flow_time = float(line_contents[3])

            if flow_time == 0.0:
                if times_acquired is None:
                    times_acquired = False
                else:
                    times_acquired = True
                    complete_trajectory(
                        trajectories,
                        Eps,
                        Ecs,
                        Qs,
                        trajectory,
                        current_Eps,
                        current_Ecs,
                        current_Q,
                        times,
                    )
                current_Eps = []
                current_Ecs = []
                if raw_Qs:
                    current_Q = []
                else:
                    current_Q = None

            current_Eps.append(float(line_contents[4]))
            current_Ecs.append(float(line_contents[6]))
            if raw_Qs:
                current_Q.append(float(line_contents[8]))
            elif t_max_for_Q is None or flow_time <= t_max_for_Q:
                current_Q = float(line_contents[8])
            if not times_acquired:
                times.append(float(line_contents[3]))

        complete_trajectory(
            trajectories,
            Eps,
            Ecs,
            Qs,
            trajectory,
            current_Eps,
            current_Ecs,
            current_Q,
            times,
        )

        assert len(trajectories) == len(Eps) == len(Ecs) == len(Qs)
        for i, (Ep, Ec) in enumerate(zip(Eps, Ecs)):
            if len(Ep) != len(times):
                breakpoint()
            assert len(Ep) == len(Ec) == len(times)
        if raw_Qs:
            for Q in zip(Qs):
                assert len(Q) == len(times)

    return (
        asarray(trajectories),
        asarray(times),
        bin_flows(Eps, bin_size),
        bin_flows(Ecs, bin_size),
        asarray(Qs),
    )


def file_is_up_to_date(filename, compare_date=None, compare_file=None):
    """Check if a particular measurement is newer than either `compare_date`,
    or the modification date of the file at `compare_file`."""

    if compare_date and compare_file:
        raise ValueError(
            "Only one of `compare_date` and `compare_file` may be specified."
        )
    if not compare_date and not compare_file:
        raise ValueError("One of `compare_date` and `compare_file` must be specified.")
    if compare_file:
        compare_date = getmtime(compare_file)
    else:
        compare_date = datetime.timestamp(compare_date)

    try:
        date_to_check = getmtime(filename)
    except OSError:
        return False
    else:
        if date_to_check > compare_date:
            return True
        else:
            return False
