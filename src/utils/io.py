"""Shared IO helpers: chunked CSV reading and tiny JSON sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd

# --------------------------------------------------------------------------- #
# Raw dataset schema
# --------------------------------------------------------------------------- #
# The group's notebook reads everything as string except post_code, which keeps
# the messy real-world values ("", ".0" suffixes, mixed date formats) parseable
# under our own control. Keep that behaviour.
TEXT_COLUMNS: tuple[str, ...] = (
    "property_id",
    "download_date",
    "council_name",
    "address",
    "post_code",
    "property_type",
    "strata_lot_number",
    "property_name",
    "area_type",
    "contract_date",
    "settlement_date",
    "zoning",
    "nature_of_property",
    "primary_purpose",
    "legal_description",
)

# price/area are read as string too: the raw file mixes blank strings and
# non-numeric junk, and coercing with errors="coerce" after the fact lets us
# count what was unusable instead of silently producing NaN dtypes.
_MESSY_NUMERIC = ("purchase_price", "area")

READ_DTYPES: dict[str, str] = {c: "string" for c in TEXT_COLUMNS + _MESSY_NUMERIC}


def iter_raw_chunks(path: Path, chunksize: int, usecols=None) -> Iterator[pd.DataFrame]:
    """Yield raw DataFrame chunks from the property CSV."""
    reader = pd.read_csv(
        path,
        dtype=READ_DTYPES,
        chunksize=chunksize,
        usecols=usecols,
        low_memory=False,
    )
    yield from reader


def count_rows(path: Path, chunksize: int) -> int:
    """Count rows without materialising the file."""
    total = 0
    for chunk in pd.read_csv(path, usecols=[0], dtype="string", chunksize=chunksize):
        total += len(chunk)
    return total


def read_postcode_key(series: pd.Series) -> pd.Series:
    """Normalise a postcode column to a 4-char zero-padded string.

    Handles the raw file's float-ish artefacts ("2112.0"), stray whitespace and
    blank strings. Returns pandas NA for unusable values.
    """
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace("", pd.NA)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    out = numeric.astype("Int64").astype("string")
    return out.str.zfill(4)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
