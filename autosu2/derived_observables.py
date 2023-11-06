from pandas import merge

from .w0 import DEFAULT_W0


def merge_and_add_mhat2(data):
    m_values = data[data.observable == "g5_mass"]
    w0_values = data[(data.observable == "w0c") & (data.free_parameter == DEFAULT_W0)]

    merged_data = merge(m_values, w0_values, on="label", suffixes=("_m", "_w0"))
    merged_data["value_mhat2"] = merged_data.value_m**2 * merged_data.value_w0**2
    merged_data["uncertainty_mhat2"] = (
        merged_data.value_m
        * merged_data.value_w0
        * (
            2
            * (
                merged_data.value_m**2 * merged_data.uncertainty_w0**2
                + merged_data.value_w0**2 * merged_data.uncertainty_m**2
            )
        )
        ** 0.5
    )
    return merged_data


def merge_no_w0(data, quantities, how="left"):
    merged_data = (
        data[data.observable == quantities[0]]
        .rename(
            columns={
                "value": f"value_{quantities[0]}",
                "uncertainty": f"uncertainty_{quantities[0]}",
            }
        )
        .drop(columns=["observable"])
    )
    for quantity in quantities[1:]:
        # Filter out desired observable
        quantity_values = data[data.observable == quantity][
            ["label", "value", "uncertainty"]
        ].rename(
            columns={
                "value": f"value_{quantity}",
                "uncertainty": f"uncertainty_{quantity}",
            }
        )

        # Join these data in
        merged_data = merge(
            merged_data,
            quantity_values,
            how=how,
            on="label",
            suffixes=("", f"_{quantity}"),
        )

    return merged_data


def merge_quantities(data, quantities, hat=False, how="left"):
    merged_data = (
        data[(data.observable == "w0c") & (data.free_parameter == DEFAULT_W0)]
        .rename(columns={"value": "value_w0", "uncertainty": "uncertainty_w0"})
        .drop(columns=["observable"])
    )
    if hat:
        merged_data["value_m_hat"] = merged_data.m * merged_data.value_w0
        merged_data["uncertainty_m_hat"] = merged_data.m * merged_data.uncertainty_w0

    for quantity in quantities:
        # Filter out desired observable
        quantity_values = data[data.observable == quantity][
            ["label", "value", "uncertainty"]
        ].rename(
            columns={
                "value": f"value_{quantity}",
                "uncertainty": f"uncertainty_{quantity}",
            }
        )

        # Join these data in
        merged_data = merge(
            merged_data,
            quantity_values,
            how=how,
            on="label",
            suffixes=("", f"_{quantity}"),
        )

        if hat:
            # Calculate hatted values
            merged_data[f"value_{quantity}_hat"] = (
                merged_data.value_w0 * merged_data[f"value_{quantity}"]
            )
            merged_data[f"uncertainty_{quantity}_hat"] = (
                merged_data.value_w0**2 * merged_data[f"uncertainty_{quantity}"] ** 2
                + merged_data.uncertainty_w0**2 * merged_data[f"value_{quantity}"] ** 2
            ) ** 0.5

            # Calculate squared hatted values
            merged_data[f"value_{quantity}_hat_squared"] = (
                merged_data[f"value_{quantity}_hat"] ** 2
            )
            merged_data[f"uncertainty_{quantity}_hat_squared"] = (
                2
                * merged_data[f"value_{quantity}_hat"]
                * merged_data[f"uncertainty_{quantity}_hat"]
            )

    return merged_data


def merge_and_hat_quantities(data, quantities):
    return merge_quantities(data, quantities, hat=True)
