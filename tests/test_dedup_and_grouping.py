"""Regression tests for de-duplication and `group_key` construction (D1/D2).

Two defects are pinned here.

1. Whole-row de-duplication is not equivalent to de-duplicating sale events.
   Rows that differ only in a column dropped later (``settlement_date``,
   ``legal_description``, ``strata_lot_number``) survive as distinct rows and
   then become byte-identical in the modelling frame, where they double-weight
   one transaction.

2. ``address|postcode`` collapses when the address has no house number. Bare
   street names (" WOODBURY PARK DR, MARDI") merged up to 62 different houses
   into one CV group, so those rows were almost never eligible for a validation
   fold.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import (
    DUPLICATE_KEY,
    add_group_key,
    drop_duplicate_rows,
    has_house_number,
)

# --------------------------------------------------------------------------- #
# 1. De-duplication
# --------------------------------------------------------------------------- #


def _sale(**over):
    row = {
        "property_id": "1001",
        "contract_date": pd.Timestamp("2015-06-01"),
        "purchase_price": 500_000.0,
        "area": 600.0,
        "area_type": "M",
        "address": "12 SMITH ST",
        "post_code": "2000",
        "settlement_date": pd.Timestamp("2015-08-01"),
        "legal_description": "LOT 1 DP 123",
        "strata_lot_number": None,
    }
    row.update(over)
    return row


def test_whole_row_dedup_misses_records_that_differ_only_in_dropped_columns():
    """The defect: these two rows are one sale, but whole-row dedup keeps both."""
    df = pd.DataFrame([_sale(), _sale(settlement_date=pd.Timestamp("2015-09-01"))])

    assert df.duplicated().sum() == 0, "fixture must not be whole-row identical"
    _, removed = drop_duplicate_rows(df)  # whole-row behaviour
    assert removed == 0

    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 1
    assert len(out) == 1


def test_identity_dedup_keeps_genuine_concurrent_sales_at_different_prices():
    """Same property, same day, *different* price = two real records, keep both."""
    df = pd.DataFrame([
        _sale(purchase_price=248_680.0),
        _sale(purchase_price=348_680.0),
    ])
    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 0
    assert len(out) == 2


def test_identity_dedup_keeps_same_price_at_different_areas():
    df = pd.DataFrame([_sale(area=600.0), _sale(area=610.0)])
    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 0
    assert set(out["area"]) == {600.0, 610.0}


def test_identity_dedup_collapses_three_four_and_two_fold_repeats():
    df = pd.DataFrame([_sale()] * 4 + [_sale(property_id="2002")] * 2)
    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 4
    assert len(out) == 2


def test_dedup_falls_back_to_whole_row_when_key_absent():
    df = pd.DataFrame([{"a": 1, "b": 2}, {"a": 1, "b": 2}, {"a": 1, "b": 3}])
    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 1
    assert len(out) == 2


def test_dedup_is_order_stable_after_later_column_selection():
    """The point of the fix: the result must still be unique in a subset frame."""
    df = pd.DataFrame([
        _sale(),
        _sale(settlement_date=pd.Timestamp("2015-09-01")),
        _sale(legal_description="LOT 2 DP 123"),
    ])
    out, removed = drop_duplicate_rows(df, subset=DUPLICATE_KEY)
    assert removed == 2
    modelling_columns = ["property_id", "contract_date", "purchase_price", "area"]
    assert out[modelling_columns].duplicated().sum() == 0


# --------------------------------------------------------------------------- #
# 2. group_key
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("address,expected", [
    ("12 SMITH ST", True),
    ("UNIT 5/12 SMITH ST", True),
    ("5/12 SMITH ST", True),
    ("LOT 3 SMITH RD", True),
    ("910 B/6 NANCARROW AVE", True),
    (" SMITH ST", False),
    ("WOODBURY PARK DR", False),
    ("", False),
    (None, False),
])
def test_has_house_number(address, expected):
    series = pd.Series([address], dtype="string")
    assert bool(has_house_number(series).iloc[0]) is expected


def test_bare_street_name_falls_back_to_property_id():
    """The defect: three different houses on one street shared a single key."""
    df = pd.DataFrame([
        _sale(property_id="9001", address=" WOODBURY PARK DR", purchase_price=299_000.0),
        _sale(property_id="9002", address=" WOODBURY PARK DR", purchase_price=400_000.0),
        _sale(property_id="9003", address=" WOODBURY PARK DR", purchase_price=535_000.0),
    ])
    out = add_group_key(df, verbose=False)
    assert out["group_key"].nunique() == 3
    assert set(out["group_key"]) == {"9001", "9002", "9003"}
    assert set(out["group_key_source"]) == {"property_id"}


def test_numbered_addresses_still_group_by_address():
    df = pd.DataFrame([
        _sale(property_id="9001", address="12 SMITH ST"),
        _sale(property_id="9002", address="14 SMITH ST"),
    ])
    out = add_group_key(df, verbose=False)
    assert set(out["group_key_source"]) == {"address"}
    assert out["group_key"].nunique() == 2
    assert all("SMITH ST" in k for k in out["group_key"])


def test_same_dwelling_sold_twice_shares_one_group_key():
    df = pd.DataFrame([
        _sale(contract_date=pd.Timestamp("2015-06-01")),
        _sale(contract_date=pd.Timestamp("2019-06-01"), purchase_price=700_000.0),
    ])
    out = add_group_key(df, verbose=False)
    assert out["group_key"].nunique() == 1


def test_bare_street_name_without_property_id_is_not_crashed():
    """No property_id column at all: fall back to the address key verbatim."""
    df = pd.DataFrame([
        _sale(address=" WOODBURY PARK DR", purchase_price=299_000.0),
        _sale(address=" WOODBURY PARK DR", purchase_price=400_000.0),
    ]).drop(columns=["property_id"])
    out = add_group_key(df, verbose=False)
    assert out["group_key"].nunique() == 1
    assert set(out["group_key_source"]) == {"address"}


def test_group_key_is_never_missing():
    df = pd.DataFrame([
        _sale(address=None, property_id="9001"),
        _sale(address="", property_id=None, post_code=None),
    ])
    out = add_group_key(df, verbose=False)
    assert out["group_key"].notna().all()
    assert (out["group_key"].astype("string").str.len() > 0).all()


# --------------------------------------------------------------------------- #
# 3. Integration: `build_clean` must actually call both fixes
# --------------------------------------------------------------------------- #
# The unit tests above pin the helpers. These pin the wiring, so that reverting
# `drop_duplicate_rows(df, subset=DUPLICATE_KEY)` to `drop_duplicate_rows(df)` --
# which the helper-level tests cannot detect -- fails here.


@pytest.fixture
def patched_build(monkeypatch):
    """Run `build_clean` over a hand-built 4-row chunk, no file I/O."""
    import src.data.clean as clean

    monkeypatch.setattr(clean, "join_transport", lambda chunk, *a, **k: chunk)

    def _run(rows: list[dict]) -> pd.DataFrame:
        chunk = pd.DataFrame(rows)
        monkeypatch.setattr(clean, "iter_raw_chunks", lambda *a, **k: iter([chunk]))
        frame, _ = clean.build_clean(progress=False)
        return frame

    return _run


def _raw(**over):
    """A row shaped like the raw extract (pre area-unit conversion)."""
    row = {
        "property_id": "1001",
        "contract_date": "2015-06-01",
        "purchase_price": "500000",
        "area": "600",
        "area_type": "M",
        "address": "12 SMITH ST",
        "post_code": "2000",
        "primary_purpose": "RESIDENCE",
        "property_type": "house",
        "dist_cbd": 5.0,
        "dist_train": 2.0,
        "dist_metro": 3.0,
        "dist_metro_new": 4.0,
    }
    row.update(over)
    return row


def test_build_clean_uses_the_identity_key(patched_build):
    """Two records that differ only in `settlement_date` must collapse to one."""
    frame = patched_build([
        _raw(address="12 SMITH ST", settlement_date="2015-08-01"),
        _raw(address="12 SMITH ST", settlement_date="2015-09-01"),
    ])
    assert len(frame) == 1, (
        "build_clean kept two copies of one sale event; it is de-duplicating on "
        "whole rows instead of the identity key")


def test_build_clean_keeps_concurrent_sales_at_different_prices(patched_build):
    frame = patched_build([
        _raw(purchase_price="248680"),
        _raw(purchase_price="348680"),
    ])
    assert len(frame) == 2


def test_build_clean_applies_the_property_id_fallback_for_bare_streets(patched_build):
    frame = patched_build([
        _raw(property_id="9001", address=" WOODBURY PARK DR", purchase_price="299000"),
        _raw(property_id="9002", address=" WOODBURY PARK DR", purchase_price="400000"),
        _raw(property_id="9003", address=" WOODBURY PARK DR", purchase_price="535000"),
    ])
    assert len(frame) == 3
    assert frame["group_key"].nunique() == 3, (
        "bare street names collapsed into one CV group; the property_id "
        "fallback is not wired into build_clean")
    assert set(frame["group_key_source"]) == {"property_id"}
