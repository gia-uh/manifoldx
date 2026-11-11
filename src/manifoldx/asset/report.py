"""What a run found: findings, their levels, and the text and JSON a shell or an agent reads."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from manifoldx.modeling.lint import LEVELS, Finding


def _number(value) -> str:
    """A finding's number as text: a float to three places, anything else as it is."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


@dataclass
class Report:
    """The findings of a run, worst first when sorted, with the exit code the command uses."""

    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warn"]

    @property
    def exit_code(self) -> int:
        """1 when any finding is an error, else 0."""
        return 1 if self.errors else 0

    def sorted(self) -> list[Finding]:
        """The findings worst first, keeping the order within a level."""
        return sorted(self.findings, key=lambda f: LEVELS.index(f.level))

    def to_json(self) -> str:
        """The findings as a JSON array, one object per finding."""
        return json.dumps([asdict(f) for f in self.findings], indent=2)

    def lines(self) -> list[str]:
        """One line per finding: ``CODE level part view: message (value) — hint``."""
        out = []
        for f in self.findings:
            where = " ".join(part for part in (f.part, f.view) if part)
            line = f"{f.code} {f.level} {where + ': ' if where else ''}{f.message}"
            if f.value is not None:
                line += f" ({_number(f.value)})"
            if f.hint:
                line += f" — {f.hint}"
            out.append(line.rstrip())
        return out

    def text(self) -> str:
        """Every finding on its own line."""
        return "\n".join(self.lines())

    def __str__(self) -> str:
        return self.text()

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self):
        return iter(self.findings)

    def extend(self, findings) -> "Report":
        """Add findings in place; returns self, so a stage may chain."""
        self.findings.extend(findings)
        return self
