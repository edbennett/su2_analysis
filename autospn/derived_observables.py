from pandas import merge


def merge_and_add_mhat2(data):
    m_values = data[data.observable == 'g5_mass']
    w0_values = data[(data.observable == 'w0c') &
                     (data.free_parameter == 0.35)]

    merged_data = merge(m_values, w0_values,
                        on='label',
                        suffixes=('_m', '_w0'))
    merged_data['value_mhat2'] = (merged_data.value_m ** 2 *
                                  merged_data.value_w0 ** 2)
    merged_data['uncertainty_mhat2'] = (
        merged_data.value_m * merged_data.value_w0 * (2 * (
            merged_data.value_m ** 2 * merged_data.uncertainty_w0 ** 2 +
            merged_data.value_w0 ** 2 * merged_data.uncertainty_m ** 2
        )) ** 0.5
    )
    return merged_data
