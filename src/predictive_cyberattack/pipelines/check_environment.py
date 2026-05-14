from __future__ import annotations

import sys

from predictive_cyberattack.environment import run_environment_checks


def main() -> None:
    checks = run_environment_checks()
    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"{status} {check.name}: {check.detail}")

    if not all(check.ok for check in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
