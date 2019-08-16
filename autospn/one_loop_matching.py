from .db import add_measurement, measurement_is_up_to_date, get_measurement

CHANNEL_COEFFICIENTS = {
    'g5': 'Zav',
    'gk': 'Zv',
    'g5gk': 'Zav'
}


def do_one_loop_matching(ensemble_descriptor,
                         channel_name, channel_parameters,
                         free_parameter=None, valence_mass=None):
    if channel_name not in CHANNEL_COEFFICIENTS:
        raise ValueError(f'No Z known for channel {channel_name}')

    Z_name = CHANNEL_COEFFICIENTS[channel_name]
    Z_coefficient = get_measurement(ensemble_descriptor, Z_name)
    bare_decay_constant = get_measurement(ensemble_descriptor,
                                          f'{channel_name}_decay_const')

    if measurement_is_up_to_date(ensemble_descriptor,
                                 f'{channel_name}_renormalised_decay_const',
                                 valence_mass=valence_mass,
                                 free_parameter=free_parameter,
                                 compare_date=Z_coefficient.updated):
        return

    renormalised_decay_const_value = (Z_coefficient.value
                                      * bare_decay_constant.value)
    renormalised_decay_const_error = (
        Z_coefficient.value ** 2 * bare_decay_constant.uncertainty ** 2
        + Z_coefficient.uncertainty ** 2 * bare_decay_constant.value ** 2
    ) ** 0.5

    add_measurement(ensemble_descriptor,
                    f'{channel_name}_renormalised_decay_const',
                    renormalised_decay_const_value,
                    renormalised_decay_const_error,
                    free_parameter=free_parameter,
                    valence_mass=valence_mass)

    return renormalised_decay_const_value, renormalised_decay_const_error
