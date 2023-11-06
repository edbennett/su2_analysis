from pandas import merge


def demornalise(data, observables):
    assert len(observables) == 2, ">2 observables not yet implemented"

    return merge(
        data[data.observable == observables[0]],
        data[data.observable == observables[1]],
        on=(
            "simulation_id",
            "label",
            "group_family",
            "group_rank",
            "representation",
            "Nf",
            "L",
            "T",
            "beta",
            "m",
            "quenched_mass",
        ),
        validate="one_to_one",
        suffixes=[f"_{observable}" for observable in observables],
    ).drop(columns=[f"observable_{observable}" for observable in observables])
