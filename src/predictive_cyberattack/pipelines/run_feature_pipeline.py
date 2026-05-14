from __future__ import annotations

import argparse

from predictive_cyberattack.config import load_config
from predictive_cyberattack.spark.io import read_model_input, write_parquet
from predictive_cyberattack.spark.prepare_features import build_feature_dataframe
from predictive_cyberattack.spark_session import build_spark_session


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PCA feature data with PySpark.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    spark = build_spark_session(config)

    try:
        source = read_model_input(spark, config)
        feature_df, preprocessing_model = build_feature_dataframe(source, config)
        write_parquet(feature_df, config.paths["pca_data"])
        print(f"Wrote PCA feature data to {config.paths['pca_data']}")

        preprocessing_pipeline_path = config.paths.get("preprocessing_pipeline")
        if preprocessing_pipeline_path:
            preprocessing_model.write().overwrite().save(preprocessing_pipeline_path)
            print(f"Wrote preprocessing pipeline to {preprocessing_pipeline_path}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
