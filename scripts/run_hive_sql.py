from __future__ import annotations

import argparse
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession


def split_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escaped = False

    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue

        for char in line:
            if escaped:
                current.append(char)
                escaped = False
                continue

            if char == "\\":
                current.append(char)
                escaped = True
                continue

            if quote:
                current.append(char)
                if char == quote:
                    quote = None
                continue

            if char in {"'", '"'}:
                current.append(char)
                quote = char
                continue

            if char == ";":
                statement = "".join(current).strip()
                if statement:
                    statements.append(statement)
                current = []
                continue

            current.append(char)

        current.append("\n")

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)

    return statements


def is_select(statement: str) -> bool:
    return statement.lstrip().lower().startswith(("select", "with", "show", "describe"))


def render_result(statement: str, dataframe: DataFrame, rows: int) -> None:
    print(f"\n-- {statement.splitlines()[0][:100]}")
    dataframe.show(rows, truncate=False)


def build_spark(app_name: str, warehouse_dir: str) -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.warehouse.dir", warehouse_dir)
        .enableHiveSupport()
        .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hive SQL files with Spark SQL Hive support.")
    parser.add_argument("--file", required=True, type=Path, help="SQL file to execute.")
    parser.add_argument(
        "--warehouse-dir",
        default="hdfs://localhost:9000/user/hive/warehouse",
        help="Spark/Hive warehouse directory.",
    )
    parser.add_argument("--app-name", default="predictive-cyberattack-hive-sql")
    parser.add_argument("--show-rows", default=50, type=int)
    args = parser.parse_args()

    statements = split_statements(args.file.read_text(encoding="utf-8"))
    if not statements:
        raise SystemExit(f"No SQL statements found in {args.file}")

    spark = build_spark(args.app_name, args.warehouse_dir)
    spark.sparkContext.setLogLevel("WARN")
    try:
        for statement in statements:
            result = spark.sql(statement)
            if is_select(statement):
                render_result(statement, result, args.show_rows)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
