#!/usr/bin/env python3

from datetime import datetime, timezone
import json
import os
import pathlib
import socket
import subprocess


def get_user():
    return environ["USERNAME"] if platform.startswith("win") else environ["USER"]


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

    metadata[
        "_comment"
    ] = "This file and all the files in this directory were generated automatically. Do not modify them; re-run the analysis workflow!"
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


def stamp_provenance(ensembles_filename):
    metadata = get_basic_metadata(ensembles_filename)
    for dirname in "final_plots", "final_tables":
        with open(pathlib.Path(dirname) / "info.json", "w") as info_file:
            info_file.write(json.dumps(metadata, sort_keys=True, indent=4))
