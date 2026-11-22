"""Every vehicle parameter publishes its valid range, and the builders enforce it."""

import re
from dataclasses import MISSING, fields

import pytest

from manifoldx.modeling.assets.bus import BusParams, build_bus, coach
from manifoldx.modeling.assets.car import CarParams, build_car
from manifoldx.modeling.assets.human import HumanParams, build_human
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
    passenger_truck,
    tanker,
)
from manifoldx.modeling.assets.vehicle import ParamsRanges

CASES = [
    (CarParams, build_car, ()),
    (BusParams, build_bus, (coach,)),
    (
        TruckParams,
        build_truck,
        (cargo_truck_1980s, box_truck, tanker, dump_truck, fire_engine, passenger_truck),
    ),
    (
        CycleParams,
        build_cycle,
        (bicycle, cycle_rickshaw, cargo_tricycle, motorcycle, electric_moto),
    ),
    (HumanParams, build_human, ()),
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
def test_each_numeric_range_end_builds_or_validate_refuses_it(cls, builder, presets):
    names = [field.name for field in fields(cls)]
    for name, spec in cls.ranges().items():
        if "min" not in spec:
            continue
        for end in ("min", "max"):
            kwargs = {name: spec[end]}
            if name != "detail":
                kwargs["detail"] = 0.5
            params = cls(**kwargs)
            try:
                builder(params)
            except ValueError as exc:
                message = str(exc)
                named = [other for other in names if other != name and re.search(rf"\b{re.escape(other)}\b", message)]
                assert named, (
                    f"{cls.__name__}.{name}={spec[end]!r} was refused without naming another field: {message}"
                )
                # A range end may only be refused by validate(), never by the builder itself.
                with pytest.raises(ValueError):
                    params.validate()
            except Exception as exc:
                raise AssertionError(
                    f"{cls.__name__}.{name}={spec[end]!r} breaks the builder: {type(exc).__name__}: {exc}"
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


def sedan(L, width, track, wheels, r, hood, cowl_u, ws_u, top, roof_end_u, deck_u, deck, tail, belt, **kw):
    """A three-box sedan from drawn ratios: heights as fractions of L, stations as fractions of L."""
    wf, wr = wheels
    headlamp_center = kw.pop("headlamp_center", (0.36 * width, (hood - 0.07) * L))
    taillamp_center = kw.pop("taillamp_center", (0.38 * width, (deck - 0.065) * L))
    return CarParams(length=L, width=width, track=track, height=top * L, wheelbase=(wr - wf) * L,
                     front_overhang=wf * L, wheel_radius=r * L, nose_top=(hood - 0.035) * L, hood_front=hood * L,
                     nose_blunt=0.85, cowl_u=cowl_u, cowl_height=(hood + 0.012) * L, windscreen_u=ws_u,
                     roof_peak_u=(ws_u + roof_end_u) / 2, roof_end_u=roof_end_u, roof_end_drop=0.01 * L,
                     hatch_u=deck_u, hatch_height=deck * L, tail_top=tail * L, belt_front=belt * L,
                     belt_rear=belt * L, sill=0.06 * L, bumper_bottom_front=0.07 * L,
                     bumper_bottom_rear=0.07 * L, hood_crown=0.03, nose_half=0.9, tail_half=0.92,
                     round_front=0.08, round_rear=0.08, tumblehome=0.14, shoulder=0.98, a_pillar_foot=0.15,
                     b_pillar_width=0.06, handles=True, crease=30.0, headlamp_center=headlamp_center,
                     taillamp_center=taillamp_center,
                     grille=kw.pop("grille", None) or (0.26 * width, (hood - 0.115) * L, (hood - 0.045) * L), **kw)


ALMENDRON = sedan(4.5, 1.80, 1.48, (0.18, 0.78), 0.073, 0.278, 0.33, 0.40, 0.396, 0.70, 0.78, 0.275, 0.262, 0.255,
                  headlamp="round_pod", headlamp_center=None, headlamp_size=(0.24, 0.24),
                  taillamp="fin", taillamp_center=None,
                  fender_crown=0.12, fin_height=0.16, fin_start_u=0.72, fin_width=0.20, windscreen_wrap=0.6,
                  side_trim=((0.78, 0.55, 0.95, "chrome"), (0.36, 0.2, 0.8, "chrome")),
                  black_pillars=True, roof_paint=True, roof_pillars=True,
                  bumper="chrome_bar", bumper_height=0.36, bumper_proud=0.06, overriders=True,
                  grille_style="mesh", grille_span="between_lamps", grille=(0.62, 0.42, 0.70),
                  wheel="hubcap", whitewall=0.05)

LADA = sedan(4.13, 1.62, 1.36, (0.17, 0.77), 0.07, 0.272, 0.31, 0.37, 0.412, 0.74, 0.80, 0.29, 0.275, 0.25,
             headlamp="rect", headlamp_size=(0.24, 0.14), taillamp="block", black_pillars=False,
             bumper="chrome_bar", bumper_height=0.35, bumper_proud=0.055,
             grille_style="mesh", grille_span="full_width", wheel="hubcap")

MOSKVICH = sedan(4.25, 1.55, 1.27, (0.17, 0.76), 0.075, 0.28, 0.31, 0.37, 0.428, 0.75, 0.80, 0.30, 0.29, 0.265,
                 headlamp="round_pod", headlamp_size=(0.16, 0.16), taillamp="block", black_pillars=False,
                 bumper="rubber", bumper_height=0.34, bumper_proud=0.05,
                 grille_style="mesh", wheel="hubcap")

POLICE = sedan(4.13, 1.62, 1.36, (0.17, 0.77), 0.07, 0.272, 0.31, 0.37, 0.412, 0.74, 0.80, 0.29, 0.275, 0.25,
               headlamp="rect", headlamp_size=(0.24, 0.14), taillamp="block", black_pillars=False,
               bumper="chrome_bar", bumper_height=0.35, bumper_proud=0.055,
               grille_style="mesh", grille_span="full_width", wheel="hubcap",
               livery=(("#1f3f8f", [(0.0, 0.55), (1.0, 0.55), (1.0, 0.85), (0.0, 0.85)]),),
               lightbar="bar")

CITY_BUS = BusParams(
    wheel="hubcap",
    livery=(("green", [(0.0, 0.26), (0.055, 0.26), (0.055, 0.88), (0.0, 0.88)]),
            ("green", [(0.72, 0.26), (1.0, 0.26), (1.0, 1.14), (0.72, 1.14)]),
            ("green", [(0.72, 2.72), (1.0, 2.72), (1.0, 3.1), (0.72, 3.1)]),
            ("cyan", [(0.0, 2.72), (0.32, 2.72), (0.40, 3.1), (0.0, 3.1)]),
            ("stripe", [(0.10, 0.98), (0.72, 0.98), (0.72, 1.14), (0.10, 1.14)])),
    front_livery=(("green", [(-1.02, 0.26), (1.02, 0.26), (1.02, 0.88), (-1.02, 0.88)]),
                  ("cyan", [(-1.02, 3.0), (1.02, 3.0), (1.02, 3.09), (-1.02, 3.09)])),
    rear_livery=(("green", [(-1.02, 0.26), (1.02, 0.26), (1.02, 1.0), (-1.02, 1.0)]),
                 ("stripe", [(-1.02, 1.04), (1.02, 1.04), (1.02, 1.15), (-1.02, 1.15)])),
    roof_livery=(("cyan", [(0.0, -1.27), (0.38, -1.27), (0.38, 1.27), (0.0, 1.27)]),
                 ("green", [(0.72, -1.27), (1.0, -1.27), (1.0, 1.27), (0.72, 1.27)])),
)

REFERENCE = [
    ("almendron", ALMENDRON, build_car),
    ("lada", LADA, build_car),
    ("moskvich", MOSKVICH, build_car),
    ("police", POLICE, build_car),
    ("city_bus", CITY_BUS, build_bus),
]


@pytest.mark.parametrize("params,builder", [case[1:] for case in REFERENCE], ids=[case[0] for case in REFERENCE])
def test_reference_parameter_sets_validate_and_build(params, builder):
    params.validate()
    builder(params)


def test_a_hatch_behind_a_pickup_bed_is_refused():
    with pytest.raises(ValueError, match="bed_u"):
        CarParams(bed_u=0.7, roof_end_u=0.66, hatch_u=0.72).validate()


def test_a_lamp_centre_above_its_face_is_refused_and_may_be_derived():
    with pytest.raises(ValueError, match="taillamp_center"):
        CarParams(hatch_height=0.9, taillamp_center=(0.66, 1.2)).validate()
    CarParams(hatch_height=0.9, taillamp_center=None).validate()
