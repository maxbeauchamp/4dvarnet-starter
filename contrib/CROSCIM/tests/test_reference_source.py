"""Tests for the generic reference-source selection (grid no longer hardcoded
to asip). Run from the repo root, e.g.:

    pytest contrib/CROSCIM/tests/test_reference_source.py
"""
import pytest

from contrib.CROSCIM.dataloaders.load_data import (
    SOURCE_RESOLUTION_M,
    effective_pixel_factor,
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
    # asip must stay the finest source for the "preferred" default and for
    # the Phase B multires-scaling formula to be a no-op when asip is active.
    assert SOURCE_RESOLUTION_M["asip"] < SOURCE_RESOLUTION_M["cimr"]
    assert SOURCE_RESOLUTION_M["asip"] < SOURCE_RESOLUTION_M["cristal"]


def test_effective_pixel_factor_is_identity_for_asip():
    # multires values are expressed in units of asip's native resolution, so
    # asip as reference must always be a pure pass-through (zero behavior
    # change for existing configs).
    for nominal in (50, 10, 2):
        assert effective_pixel_factor(nominal, "asip") == nominal


def test_effective_pixel_factor_converts_for_coarser_reference():
    # cimr's native resolution (5000m) is 10x asip's (500m): a nominal "50"
    # (-> 25km target) becomes a raw factor of 5 on cimr's own native pixels.
    assert effective_pixel_factor(50, "cimr") == 5
    # a nominal "10" (-> 5km target) exactly matches cimr's native
    # resolution: no further coarsening needed (factor 1).
    assert effective_pixel_factor(10, "cimr") == 1


def test_effective_pixel_factor_rejects_unreachable_target():
    # nominal "2" (-> 1km target) is finer than cimr's 5km native resolution
    # — physically impossible, must raise rather than silently misbehave.
    with pytest.raises(ValueError):
        effective_pixel_factor(2, "cimr")


@pytest.mark.parametrize("source,multires", [("asip", [50, 10, 2]), ("cimr", [50, 10]), ("cristal", [50, 10])])
def test_effective_pixel_factor_preserves_ratios(source, multires):
    # data_multires.py/data_multires_supervised.py reuse the SAME `factor`
    # ratio (e.g. nominal_level // nominal_resize) both in nominal space
    # (enlarged_dims lookups, mask coarsening) and to derive the real
    # coarsening factor via effective_pixel_factor. This only stays correct
    # if ratios between consecutive nominal levels equal the ratios between
    # their effective counterparts — verify that invariant holds.
    effective = [effective_pixel_factor(m, source) for m in multires]
    for i in range(len(multires) - 1):
        assert multires[i] / multires[i + 1] == pytest.approx(effective[i] / effective[i + 1])
