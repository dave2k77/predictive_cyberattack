from __future__ import annotations

import importlib.metadata
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_environment_checks() -> list[CheckResult]:
    return [
        _check_java(),
        _check_pyspark(),
    ]


def _check_java() -> CheckResult:
    java_path = shutil.which("java")
    if not java_path:
        return CheckResult("java", False, "Java was not found on PATH. Install OpenJDK 17.")

    process = subprocess.run(
        ["java", "-version"],
        check=False,
        capture_output=True,
        text=True,
    )
    version_output = process.stderr or process.stdout
    version = _parse_java_major_version(version_output)
    if version is None:
        return CheckResult("java", False, f"Could not parse Java version from: {version_output.strip()}")

    if version < 17:
        return CheckResult("java", False, f"Java {version} is too old. Use OpenJDK 17.")

    if version > 21:
        return CheckResult(
            "java",
            False,
            f"Java {version} is too new for this Spark/Hadoop stack. Use OpenJDK 17.",
        )

    return CheckResult("java", True, f"{java_path} reports Java {version}")


def _check_pyspark() -> CheckResult:
    try:
        version = importlib.metadata.version("pyspark")
    except importlib.metadata.PackageNotFoundError:
        return CheckResult("pyspark", False, "PySpark is not installed. Run `make install`.")

    return CheckResult("pyspark", True, f"PySpark {version}")


def _parse_java_major_version(version_output: str) -> int | None:
    match = re.search(r'version "([0-9]+)(?:\.([0-9]+))?', version_output)
    if not match:
        return None

    first = int(match.group(1))
    if first == 1 and match.group(2):
        return int(match.group(2))

    return first
