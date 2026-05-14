from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    raw: dict[str, Any]

    @property
    def app_name(self) -> str:
        return str(self.raw.get("app_name", "predictive-cyberattack"))

    @property
    def mode(self) -> str:
        return str(self.raw.get("mode", "demo"))

    @property
    def master(self) -> str | None:
        master = self.raw.get("master")
        return None if master in (None, "null") else str(master)

    @property
    def enable_hive(self) -> bool:
        return bool(self.raw.get("enable_hive", False))

    @property
    def paths(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.raw.get("paths", {}).items()}

    @property
    def features(self) -> dict[str, Any]:
        return dict(self.raw.get("features", {}))

    @property
    def training(self) -> dict[str, Any]:
        return dict(self.raw.get("training", {}))


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        return ProjectConfig(yaml.safe_load(config_file) or {})
