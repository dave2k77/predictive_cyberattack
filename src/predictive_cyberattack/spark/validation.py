from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from predictive_cyberattack.config import ProjectConfig


class DataValidationError(ValueError):
    """Raised when source data does not satisfy the configured feature contract."""


def required_input_columns(config: ProjectConfig) -> list[str]:
    features = config.features
    columns = [
        *features.get("categorical", []),
        *features.get("numeric", []),
        features.get("label", "label"),
    ]
    return list(dict.fromkeys(columns))


def validate_input_dataframe(df: DataFrame, config: ProjectConfig) -> None:
    required_columns = required_input_columns(config)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Input data is missing required columns: {', '.join(missing_columns)}")

    if df.limit(1).count() == 0:
        raise DataValidationError("Input data is empty.")

    label_col = config.features.get("label", "label")
    invalid_labels = (
        df.select(label_col)
        .where(F.col(label_col).isNull() | ~F.col(label_col).isin(0, 1))
        .limit(10)
        .collect()
    )
    if invalid_labels:
        values = [str(row[label_col]) for row in invalid_labels]
        raise DataValidationError(f"Binary label column `{label_col}` must contain only 0 or 1. Found: {values}")

    non_numeric_columns = []
    for column in config.features.get("numeric", []):
        invalid_count = df.where(F.col(column).isNotNull() & F.col(column).cast("double").isNull()).limit(1).count()
        if invalid_count:
            non_numeric_columns.append(column)

    if non_numeric_columns:
        raise DataValidationError(
            "Numeric feature columns contain non-numeric values: " + ", ".join(non_numeric_columns)
        )


def normalize_input_schema(df: DataFrame, config: ProjectConfig) -> DataFrame:
    normalized = df
    for column in config.features.get("numeric", []):
        normalized = normalized.withColumn(column, F.col(column).cast("double"))

    label_col = config.features.get("label", "label")
    return normalized.withColumn(label_col, F.col(label_col).cast("int"))
