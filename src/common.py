"""Atomic outputs and CLI helpers; never write into the dataset."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    atomic_write(path, lambda f: f.write(json.dumps(value, indent=2, allow_nan=False).encode()))


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_npy(path, array):
    atomic_write(path, lambda stream: np.save(stream, array, allow_pickle=False))


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_output(root, split, sample_id):
    # Hash disambiguates IDs; no source path can escape output root.
    return Path(root) / "predictions" / split / hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def data_arguments(parser):
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--scene", action="append", help="Exact scene name; repeat for several scenes")
    parser.add_argument("--subset", choices=["balanced", "unbalanced"])
    parser.add_argument("--max-samples", type=positive_int)


def selected_indices(dataset, maximum):
    return range(min(len(dataset), maximum or len(dataset)))


def protect_dataset(data_root, output_dir):
    data, output = Path(data_root).resolve(), Path(output_dir).resolve()
    if output == data or data in output.parents:
        raise ValueError("Output must be outside the read-only dataset root")
