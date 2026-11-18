"""Procedural assets: objects built from named, measurable parameters (cars first, every vehicle built alike)."""

from manifoldx.modeling.assets.bus import Bus, BusParams, build_bus
from manifoldx.modeling.assets.car import Car, CarParams, Part, build_car
from manifoldx.modeling.assets.truck import (
    Truck,
    TruckParams,
    box_truck,
    build_truck,
    cargo_truck_1980s,
    dump_truck,
    fire_engine,
    tanker,
)
from manifoldx.modeling.assets.vehicle import Vehicle

__all__ = [
    "Bus", "BusParams", "Car", "CarParams", "Part", "Truck", "TruckParams", "Vehicle",
    "box_truck", "build_bus", "build_car", "build_truck", "cargo_truck_1980s", "dump_truck", "fire_engine", "tanker",
]
