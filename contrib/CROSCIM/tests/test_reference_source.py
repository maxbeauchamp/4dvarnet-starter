"""Tests for the generic reference-source selection (grid no longer hardcoded
to asip) and the static grid-reference file path helper. Run from the repo
root, e.g.:

    pytest contrib/CROSCIM/tests/test_reference_source.py
"""
import pytest

from contrib.CROSCIM.dataloaders.load_data import (
    SOURCE_RESOLUTION_M,
    gridref_path,
    resolve_reference_source,
)


def test_asip_preferred_when_present():
    assert resolve_reference_source(["cimr", "cristal", "asip"]) == "asip"
    assert resolve_reference_source(["asip"]) == "asip"


def test_cimr_wins_tiebreak_when_asip_absent():
    assert resolve_reference_source(["cimr", "cristal"]) == "cimr"


def test_falls_back_to_cristal_when_only_cristal_active():
    assert resolve_reference_source(["cristal"]) == "cristal"


def test_falls_back_to_cimr_when_only_cimr_active():
    assert resolve_reference_source(["cimr"]) == "cimr"


def test_raises_when_no_known_resolution_source_active():
    with pytest.raises(ValueError):
        resolve_reference_source(["models"])


def test_override_wins_even_when_asip_present():
    assert resolve_reference_source(["asip", "cimr"], override="cimr") == "cimr"


def test_override_must_be_active():
    with pytest.raises(ValueError):
        resolve_reference_source(["cimr"], override="cristal")


def test_source_resolution_table_has_expected_values():
    assert SOURCE_RESOLUTION_M["asip"] < SOURCE_RESOLUTION_M["cimr"]
    assert SOURCE_RESOLUTION_M["asip"] < SOURCE_RESOLUTION_M["cristal"]


def test_gridref_path_uses_default_dir_and_naming_convention():
    path = gridref_path(50)
    assert path.endswith("gridref_x50.nc")
    assert "gridref" in path


def test_gridref_path_honors_explicit_dir():
    assert gridref_path(10, gridref_dir="/tmp/somewhere") == "/tmp/somewhere/gridref_x10.nc"
