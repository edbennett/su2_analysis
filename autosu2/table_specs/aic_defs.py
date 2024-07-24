#!/usr/bin/env python3

from format_multiple_errors import format_multiple_errors

from ..derived_observables import merge_no_w0
from ..provenance import latex_metadata, get_basic_metadata, number_to_latex


def generate(data, ensembles):
    definition_filename = "assets/definitions/gammastar_aic.tex"

    merged_data = merge_no_w0(data, ["gamma_aic", "gamma_aic_syst"])
    with open(definition_filename, "w") as f:
        print(latex_metadata(get_basic_metadata(ensembles["_filename"])), file=f)
        for ensemble in merged_data.to_dict("records"):
            latex_var_name = f"GammaStarAIC{number_to_latex(ensemble['label'], tolerate_non_numbers=True)}"
            gammastar = format_multiple_errors(
                ensemble["value_gamma_aic"],
                ensemble["uncertainty_gamma_aic"],
                ensemble["value_gamma_aic_syst"],
                abbreviate=True,
                latex=True,
            )
            print(f"\\newcommand \\{latex_var_name} {{{gammastar}}}", file=f)
