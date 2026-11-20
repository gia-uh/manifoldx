"""The micro case harness now lives in manifoldx.bench.micro; this name stays for the suites."""

import sys

from manifoldx.bench import micro

sys.modules[__name__] = micro
