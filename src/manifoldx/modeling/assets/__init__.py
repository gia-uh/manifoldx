"""Procedural assets: objects built from named, measurable parameters (cars first, every vehicle built alike)."""

from manifoldx.modeling.assets.bus import Bus, BusParams, build_bus, coach
from manifoldx.modeling.assets.car import Car, CarParams, Part, build_car
from manifoldx.modeling.assets.cycle import (
    Cycle,
    CycleParams,
    bicycle,
    build_cycle,
    cargo_tricycle,
    cycle_rickshaw,
    electric_moto,
    motorcycle,
)
from manifoldx.modeling.assets.truck import (
    Truck,
    TruckParams,
    box_truck,
    build_truck,
    cargo_truck_1980s,
    dump_truck,
    fire_engine,
    passenger_truck,
    tanker,
)
from manifoldx.modeling.assets.vehicle import Vehicle

__all__ = [
    "Bus", "BusParams", "Car", "CarParams", "Cycle", "CycleParams", "Part", "Truck", "TruckParams", "Vehicle",
    "bicycle", "box_truck", "build_bus", "build_car", "build_cycle", "build_truck", "cargo_tricycle",
    "cargo_truck_1980s", "coach", "cycle_rickshaw", "dump_truck", "electric_moto", "fire_engine", "motorcycle",
    "passenger_truck", "tanker",
]
