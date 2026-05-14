from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession

from predictive_cyberattack.config import ProjectConfig


def read_model_input(spark: SparkSession, config: ProjectConfig) -> DataFrame:
    paths = config.paths

    if config.mode == "demo":
        return spark.read.csv(paths["raw_csv"], header=True, inferSchema=True)

    source_table = paths.get("source_table") or paths.get("cleaned_table")
    if source_table:
        return spark.table(source_table)

    return spark.read.parquet(paths["model_input"])


def write_parquet(df: DataFrame, path: str) -> None:
    df.write.mode("overwrite").parquet(path)


def ensure_local_dir(path: str) -> Path:
    local_path = Path(path)
    local_path.mkdir(parents=True, exist_ok=True)
    return local_path
