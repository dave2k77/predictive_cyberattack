from __future__ import annotations

import argparse

from predictive_cyberattack.config import load_config
from predictive_cyberattack.spark.evaluate import write_metrics
from predictive_cyberattack.spark.train import train_binary_models, train_multiclass_models
from predictive_cyberattack.spark_session import build_spark_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PySpark ML classifiers.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to a YAML config file.")
    parser.add_argument("--task", choices=("binary", "multiclass", "all"), default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    spark = build_spark_session(config)

    try:
        pca_data = spark.read.parquet(config.paths["pca_data"])
        if args.task in ("binary", "all"):
            binary_metrics = train_binary_models(pca_data, config)
            output_path = write_metrics(binary_metrics, config.paths["metrics_dir"], "binary_metrics.json")
            print(f"Wrote binary metrics to {output_path}")

        if args.task in ("multiclass", "all"):
            multiclass_metrics = train_multiclass_models(pca_data, config)
            output_path = write_metrics(multiclass_metrics, config.paths["metrics_dir"], "multiclass_metrics.json")
            print(f"Wrote multiclass metrics to {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
