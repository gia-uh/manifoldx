"""Benchmark, profile and check any manifoldx app: scenarios, probes, run folders and micro cases.

`manifoldx bench` is the command; this package is what an app's scenarios and micro cases import.
"""

from manifoldx.bench import micro
from manifoldx.bench.micro import case

__all__ = ["case", "micro"]
