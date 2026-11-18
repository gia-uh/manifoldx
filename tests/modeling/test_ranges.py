"""Every vehicle parameter publishes its valid range, and the builders enforce it."""

from dataclasses import MISSING, fields

import pytest

from manifoldx.modeling.assets.bus import BusParams, build_bus
from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.assets.cycle import (
    CycleParams,
    bicycle,
    build_cycle,
    cargo_tricycle,
    cycle_rickshaw,
    electric_moto,
    motorcycle,
)
from manifoldx.modeling.assets.truck import (
    TruckParams,
    box_truck,
    build_truck,
    cargo_truck_1980s,
    dump_truck,
    fire_engine,
    tanker,
)
from manifoldx.modeling.assets.vehicle import ParamsRanges

CASES = [
    (CarParams, build_car, ()),
    (BusParams, build_bus, ()),
    (
        TruckParams,
        build_truck,
        (cargo_truck_1980s, box_truck, tanker, dump_truck, fire_engine),
    ),
    (
        CycleParams,
        build_cycle,
        (bicycle, cycle_rickshaw, cargo_tricycle, motorcycle, electric_moto),
    ),
]
IDS = [case[0].__name__ for case in CASES]


def _default(field):
    if field.default is not MISSING:
        return field.default
    return field.default_factory()


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_the_params_share_one_range_base(cls, builder, presets):
    assert issubclass(cls, ParamsRanges)


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_every_field_has_a_range_choices_or_kind(cls, builder, presets):
    specs = cls.ranges()
    assert set(specs) == {field.name for field in fields(cls)}
    for field in fields(cls):
        spec = specs[field.name]
        assert spec["default"] == _default(field), field.name
        if "min" in spec or "max" in spec:
            assert isinstance(spec["min"], (int, float)), field.name
            assert isinstance(spec["max"], (int, float)), field.name
            assert spec["min"] < spec["max"], field.name
            assert "step" in spec and "unit" in spec, field.name
        elif "choices" in spec:
            assert len(spec["choices"]) >= 1, field.name
        else:
            assert spec["kind"] in {"tuple", "shape", "list", "colour", "string"}, field.name


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_every_default_and_preset_validates(cls, builder, presets):
    cls().validate()
    for preset in presets:
        preset().validate()


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_each_numeric_range_end_builds(cls, builder, presets):
    for name, spec in cls.ranges().items():
        if "min" not in spec:
            continue
        for end in ("min", "max"):
            kwargs = {name: spec[end]}
            if name != "detail":
                kwargs["detail"] = 0.5
            try:
                builder(cls(**kwargs))
            except Exception as exc:
                raise AssertionError(
                    f"{cls.__name__}.{name}={spec[end]!r} does not build: {exc}"
                ) from exc


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_a_number_outside_its_range_raises_naming_the_field(cls, builder, presets):
    for name, spec in cls.ranges().items():
        if "min" not in spec:
            continue
        for value in (spec["min"] - 0.1, spec["max"] + 0.1):
            with pytest.raises(ValueError, match=name):
                cls(**{name: value}).validate()


@pytest.mark.parametrize("cls,builder,presets", CASES, ids=IDS)
def test_an_unknown_choice_raises_naming_the_field(cls, builder, presets):
    for name, spec in cls.ranges().items():
        if "choices" not in spec:
            continue
        value = "__not_a_choice__"
        assert value not in spec["choices"]
        with pytest.raises(ValueError, match=name):
            cls(**{name: value}).validate()


def test_a_car_whose_axles_do_not_fit_its_length_is_refused():
    with pytest.raises(ValueError, match="wheelbase"):
        CarParams(length=3.6, wheelbase=3.0, front_overhang=0.8).validate()


def test_a_car_whose_sill_is_above_its_belt_is_refused():
    with pytest.raises(ValueError, match="sill"):
        CarParams(sill=0.9, belt_front=0.8).validate()


def test_a_car_whose_profile_fractions_are_out_of_order_is_refused():
    with pytest.raises(ValueError, match="cowl_u"):
        CarParams(cowl_u=0.32, windscreen_u=0.30).validate()


def test_a_pickup_may_start_its_bed_before_the_closed_hatch():
    CarParams(bed_u=0.7, roof_end_u=0.66, hatch_u=0.68).validate()


def test_a_bus_door_past_its_end_is_refused():
    with pytest.raises(ValueError, match="doors"):
        BusParams(doors=((0.95, 1.2, 2),)).validate()


def test_bus_doors_that_overlap_are_refused():
    with pytest.raises(ValueError, match="doors"):
        BusParams(doors=((0.2, 1.2, 2), (0.25, 1.2, 2))).validate()


def test_a_truck_cab_longer_than_its_length_is_refused():
    with pytest.raises(ValueError, match="cab_length"):
        TruckParams(length=6.0, cab_length=7.0).validate()


def test_a_cycle_whose_wheels_overlap_is_refused():
    with pytest.raises(ValueError, match="wheelbase"):
        CycleParams(wheelbase=0.7, wheel_radius=0.4).validate()


def test_a_cycle_layout_that_disagrees_with_its_wheel_count_is_refused():
    with pytest.raises(ValueError, match="layout"):
        CycleParams(wheels=3, layout="inline").validate()


def test_a_cycle_canopy_without_a_payload_is_refused():
    with pytest.raises(ValueError, match="canopy"):
        CycleParams(canopy=True, payload="none").validate()
