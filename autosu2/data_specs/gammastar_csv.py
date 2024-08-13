#!/usr/bin/env python3

import numpy as np
import pandas as pd

from ..provenance import text_metadata, get_basic_metadata


def generate(data, ensembles):
    filename = "gammastar_results.csv"
    common_metadata = {
        "group_family": "SUN",
        "group_rank": 2,
        "representation": "ADJ",
    }

    finitebeta_data = pd.read_csv("processed_data/gammastar_fshs.csv", comment="#")
    contlim_data = pd.read_csv("processed_data/gammastar_contlim.csv", comment="#")
    contlim_data["beta"] = np.inf

    combined_data = pd.concat([finitebeta_data, contlim_data])
    for key, value in reversed(common_metadata.items()):
        combined_data.insert(0, key, value)

    with open(filename, "w") as f:
        print(text_metadata(get_basic_metadata(ensembles["_filename"])), file=f)
        combined_data.to_csv(f, index=False)
