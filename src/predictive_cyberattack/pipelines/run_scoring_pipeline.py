from __future__ import annotations

import argparse

from pyspark.ml.classification import (
    DecisionTreeClassificationModel,
    GBTClassificationModel,
    LogisticRegressionModel,
    NaiveBayesModel,
    RandomForestClassificationModel,
)
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import functions as F

from predictive_cyberattack.config import load_config
from predictive_cyberattack.spark.io import read_model_input
from predictive_cyberattack.spark_session import build_spark_session


BINARY_MODELS = {
    "gradient_boosted_tree": GBTClassificationModel,
    "logistic_regression": LogisticRegressionModel,
    "naive_bayes": NaiveBayesModel,
}

MULTICLASS_MODELS = {
    "decision_tree": DecisionTreeClassificationModel,
    "naive_bayes": NaiveBayesModel,
    "random_forest": RandomForestClassificationModel,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score records with persisted PySpark preprocessing and models.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to a YAML config file.")
    parser.add_argument("--output", help="Output parquet path. Defaults to paths.scored_predictions.")
    parser.add_argument("--binary-model", default="gradient_boosted_tree", choices=sorted(BINARY_MODELS))
    parser.add_argument("--multiclass-model", default="random_forest", choices=sorted(MULTICLASS_MODELS))
    parser.add_argument("--limit", type=int, help="Optional row limit for quick demonstrations.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.paths
    output_path = args.output or paths.get("scored_predictions")
    if not output_path:
        raise SystemExit("Provide --output or set paths.scored_predictions in the config.")

    spark = build_spark_session(config)
    try:
        source = read_model_input(spark, config)
        if args.limit:
            source = source.limit(args.limit)

        preprocessing = PipelineModel.load(paths["preprocessing_pipeline"])
        features = preprocessing.transform(source).withColumn("_score_row_id", F.monotonically_increasing_id())

        binary_path = f"{paths['models_dir']}/binary/{args.binary_model}"
        binary_model = BINARY_MODELS[args.binary_model].load(binary_path)
        binary_predictions = (
            binary_model.transform(features)
            .select(
                "_score_row_id",
                F.col("prediction").cast("int").alias("binary_prediction"),
                F.col("probability").alias("binary_probability"),
            )
        )

        multiclass_path = f"{paths['models_dir']}/multiclass/{args.multiclass_model}"
        multiclass_model = MULTICLASS_MODELS[args.multiclass_model].load(multiclass_path)
        multiclass_predictions = (
            multiclass_model.transform(features)
            .select(
                "_score_row_id",
                F.col("prediction").cast("int").alias("attack_category_prediction"),
            )
        )

        scored = (
            features.join(binary_predictions, on="_score_row_id")
            .join(multiclass_predictions, on="_score_row_id")
            .drop("_score_row_id", "features", "scaledFeatures", "pcaFeatures")
        )
        scored.write.mode("overwrite").parquet(output_path)
        print(f"Wrote scored predictions to {output_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
