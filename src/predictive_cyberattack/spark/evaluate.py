from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame

from predictive_cyberattack.spark.io import ensure_local_dir


def evaluate_binary(predictions: DataFrame, label_col: str = "label") -> dict[str, float]:
    roc = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    ).evaluate(predictions)
    pr = BinaryClassificationEvaluator(
        labelCol=label_col,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    ).evaluate(predictions)
    accuracy = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="accuracy",
    ).evaluate(predictions)
    f1 = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="f1",
    ).evaluate(predictions)
    precision = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedPrecision",
    ).evaluate(predictions)
    recall = MulticlassClassificationEvaluator(
        labelCol=label_col,
        predictionCol="prediction",
        metricName="weightedRecall",
    ).evaluate(predictions)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "weighted_precision": precision,
        "weighted_recall": recall,
        "area_under_roc": roc,
        "area_under_pr": pr,
    }


def evaluate_multiclass(predictions: DataFrame, label_col: str = "attack_cat_index") -> dict[str, float]:
    metrics = {}
    for metric_name in ("accuracy", "f1", "weightedPrecision", "weightedRecall"):
        value = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric_name,
        ).evaluate(predictions)
        metrics[_snake_case(metric_name)] = value
    return metrics


def write_metrics(metrics: dict[str, Any], metrics_dir: str, filename: str) -> Path:
    output_dir = ensure_local_dir(metrics_dir)
    output_path = output_dir / filename
    with output_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    return output_path


def _snake_case(value: str) -> str:
    chars = []
    for char in value:
        if char.isupper():
            chars.extend(["_", char.lower()])
        else:
            chars.append(char)
    return "".join(chars).lstrip("_")
