"""Immutable diagnostic findings and report results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DoctorLevel = Literal["ok", "info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class DoctorFinding:
    level: DoctorLevel
    message: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    profile_name: str
    findings: tuple[DoctorFinding, ...]

    @property
    def errors(self) -> tuple[DoctorFinding, ...]:
        return tuple(finding for finding in self.findings if finding.level == "error")

    @property
    def warnings(self) -> tuple[DoctorFinding, ...]:
        return tuple(finding for finding in self.findings if finding.level == "warning")

    @property
    def exit_code(self) -> int:
        return 1 if self.errors else 0
