#!/usr/bin/env python3

from datetime import datetime, timezone
import json
import os
import pathlib
import socket
import subprocess


def get_commit_id():
    return (
        subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )


def get_basic_metadata(ensembles_filename):
    metadata = {}
    ensembles_file = pathlib.Path(ensembles_filename)

    metadata["_comment"] = (
        "This file and all the files in this directory were generated automatically. Do not modify them; re-run the analysis workflow!"
    )
    metadata["workflow_run"] = {
        "completed": datetime.now(timezone.utc).isoformat(),
        "user_name": os.getlogin(),
        "machine_name": socket.gethostname(),
    }
    metadata["ensembles_metadata"] = {
        "last_updated": datetime.fromtimestamp(
            ensembles_file.stat().st_mtime, timezone.utc
        ).isoformat()
    }
    metadata["analysis_code"] = {"version": get_commit_id()}

    return metadata


def flatten_metadata(metadata):
    flat_metadata = {
        "_comment": "This file was generated automatically. Do not modify it directly; re-run the analysis workflow!"
    }

    for outer_key, inner_metadata in metadata.items():
        if outer_key == "_comment":
            continue
        flat_metadata.update(
            {f"{outer_key}::{k}": v for k, v in inner_metadata.items()}
        )

    flat_metadata["generated"] = flat_metadata["workflow_run::completed"]
    del flat_metadata["workflow_run::completed"]
    return flat_metadata


def text_metadata(metadata, comment_char="#"):
    return "\n".join(
        [
            f"{comment_char} {k}: {v.replace('\n', '\n{comment_char} ')}"
            for k, v in flatten_metadata(metadata).items()
        ]
    )


number_names = {
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Nine",
    "0": "Zero",
    ".": "Point",
}


def number_to_latex(number, tolerate_non_numbers=False):
    result = str(number)
    if not tolerate_non_numbers:
        if not set(result) < set(number_names):
            raise ValueError(
                "Non-numeric characters found but tolerate_non_numbers disabled"
            )

    for char, substitution in number_names.items():
        result = result.replace(char, substitution)
    return result


def stamp_provenance(ensembles_filename):
    metadata = get_basic_metadata(ensembles_filename)
    with open("assets/info.json", "w") as info_file:
        info_file.write(json.dumps(metadata, sort_keys=True, indent=4))
