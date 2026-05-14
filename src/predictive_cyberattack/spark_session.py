from __future__ import annotations

from pyspark.sql import SparkSession

from predictive_cyberattack.config import ProjectConfig


def build_spark_session(config: ProjectConfig) -> SparkSession:
    builder = SparkSession.builder.appName(config.app_name)

    if config.master:
        builder = builder.master(config.master)

    if config.enable_hive:
        builder = builder.enableHiveSupport()

    return builder.getOrCreate()
