from collections import namedtuple, defaultdict
from uncertainties import ufloat

from .provenance import text_metadata, get_basic_metadata

HLINE = r"    \hline"
ObservableSpec = namedtuple(
    "ObservableSpec", ("name", "valence_mass", "free_parameter"), defaults=(None, None)
)
SMALLEST_RELATIVE_UNCERTAINTY = 1e-12


def table_row(row_content):
    if isinstance(row_content, str):
        return "    " + row_content
    else:
        return "    " + " & ".join(row_content)


def format_value_and_error(value, error, error_digits=2, exponential=False):
    if exponential:
        exp_flag = "e"
    else:
        exp_flag = "f"

    value = ufloat(value, error)
    format_string = f"${{:.{error_digits}u{exp_flag}SL}}$"
    return format_string.format(value)


def generate_table_from_content(
    filename,
    table_content,
    columns=None,
    header=None,
    table_spec=None,
    preamble="",
):
    if columns is not None and (header is not None or table_spec is not None):
        raise ValueError(
            "Either `columns` or `header` + `table_spec` may be " "specified, not both."
        )
    if columns:
        header = [column for column in columns if column is not None]
        table_spec = "".join(["c" if column is not None else "|" for column in columns])
    else:
        if table_spec is None:
            raise ValueError("Either `columns` or `table_spec` must be specified.")

    with open("assets/tables/" + filename, "w") as f:
        print(preamble, file=f)
        print(r"\begin{tabular}{" + table_spec + "}", file=f)
        if header:
            print(table_row(header) + r" \\", file=f)
        print(HLINE, file=f)
        print(HLINE, file=f)
        print((r" \\" "\n").join(table_content), file=f)
        print(r"\end{tabular}", file=f)


def measurement_mask(data, observable):
    measurement_mask = data.observable == observable.name
    if observable.free_parameter is None:
        measurement_mask = measurement_mask & data.free_parameter.isnull()
    else:
        measurement_mask = measurement_mask & (
            data.free_parameter == observable.free_parameter
        )
    if observable.valence_mass is None:
        measurement_mask = measurement_mask & data.valence_mass.isnull()
    else:
        measurement_mask = measurement_mask & (
            data.valence_mass == observable.valence_mass
        )
    return measurement_mask


def generate_table_from_db(
    data,
    ensemble_names,
    observables,
    filename,
    ensembles_filename,
    columns=None,
    constants=tuple(),
    error_digits=2,
    exponential=False,
    multirow=defaultdict(bool),
    header=None,
    table_spec=None,
    suppress_zeroes=False,
    skip_empty_rows=False,
):
    table_content = []
    line_content = ""

    if "V" in constants and "V" not in data.columns:
        data = data.copy()
        data["V"] = [f"{T} \\times {L}^3" for T, L in zip(data["T"], data["L"])]

    # Set up initial values of variables used for implementing multirow
    current_row_constants = {}
    num_rows = {}
    for ensemble in ensemble_names:
        if not ensemble:
            line_content += HLINE + "\n"
            continue

        ensemble_data = data[data.label == ensemble]
        if len(ensemble_data) == 0:
            print(f"WARNING: No data available for ensemble {ensemble}, " "skipping")
            continue

        row_content = [ensemble]
        previous_row_constants = current_row_constants
        current_row_constants = {}
        for constant in constants:
            value = set(ensemble_data[constant])
            assert len(value) == 1
            (value,) = value
            current_row_constants[constant] = value
            if not multirow[constant]:
                row_content.append(f"${str(value)}$")
            elif value != previous_row_constants.get(constant, None):
                # New value: finish previous multirow, start a multirow
                # and reset row count
                table_content = [
                    line.replace(f"NUM_ROWS_{constant}", str(num_rows[constant]))
                    for line in table_content
                ]
                row_content.append(
                    r"\multirow{NUM_ROWS_"
                    + constant
                    + "}{*}"
                    + "{"
                    + f"${str(value)}$"
                    + r"}"
                )
                num_rows[constant] = 1
            else:
                # Same value: blank cell, increment row count
                row_content.append("")
                num_rows[constant] += 1

        observable_found = False
        for observable in observables:
            if isinstance(observable, str):
                observable = ObservableSpec(observable)
            assert type(observable) is ObservableSpec

            measurement = ensemble_data[measurement_mask(ensemble_data, observable)]
            if len(measurement) > 1:
                raise ValueError(
                    "ensemble-observable combination is not unique for "
                    f"{ensemble} {observable}"
                )

            if len(measurement) == 0:
                row_content.append("---")
                continue
            else:
                observable_found = True

            if measurement.value.iloc[0] > 0 and (
                measurement.uncertainty.iloc[0] / measurement.value.iloc[0]
                < SMALLEST_RELATIVE_UNCERTAINTY
            ):
                print("WARNING: very small uncertainty found", measurement)
                uncertainty = 0
            else:
                uncertainty = measurement.uncertainty.iloc[0]

            if (measurement.value.iloc[0] == 0) and suppress_zeroes:
                row_content.append(r"\ll 1")
            else:
                row_content.append(
                    format_value_and_error(
                        measurement.value.iloc[0],
                        uncertainty,
                        error_digits=error_digits,
                        exponential=exponential,
                    )
                )
        if skip_empty_rows and not observable_found:
            continue

        line_content += table_row(row_content)
        table_content.append(line_content)
        line_content = ""

    if multirow:
        for constant in constants:
            if multirow[constant]:
                table_content = [
                    line.replace(f"NUM_ROWS_{constant}", str(num_rows[constant]))
                    for line in table_content
                ]

    preamble = text_metadata(get_basic_metadata(ensembles_filename), comment_char="%")
    generate_table_from_content(
        filename,
        table_content,
        columns=columns,
        header=header,
        table_spec=table_spec,
        preamble=preamble,
    )
