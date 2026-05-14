from __future__ import annotations

from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.sql import DataFrame

from predictive_cyberattack.config import ProjectConfig
from predictive_cyberattack.spark.evaluate import evaluate_binary, evaluate_multiclass


def split_data(df: DataFrame, config: ProjectConfig) -> tuple[DataFrame, DataFrame]:
    training = config.training
    train_fraction = float(training.get("train_fraction", 0.7))
    seed = int(training.get("seed", 25))
    return df.randomSplit([train_fraction, 1.0 - train_fraction], seed=seed)


def train_binary_models(df: DataFrame, config: ProjectConfig) -> dict[str, dict[str, float]]:
    train, test = split_data(df.select("pcaFeatures", "label"), config)
    models_dir = config.paths.get("models_dir")
    models = {
        "logistic_regression": LogisticRegression(featuresCol="pcaFeatures", labelCol="label"),
        "naive_bayes": NaiveBayes(featuresCol="pcaFeatures", labelCol="label", modelType="gaussian"),
        "gradient_boosted_tree": GBTClassifier(featuresCol="pcaFeatures", labelCol="label"),
    }

    results = {}
    for name, estimator in models.items():
        fitted = estimator.fit(train)
        predictions = fitted.transform(test)
        results[name] = evaluate_binary(predictions)
        if models_dir:
            fitted.write().overwrite().save(f"{models_dir}/binary/{name}")
    return results


def train_multiclass_models(df: DataFrame, config: ProjectConfig) -> dict[str, dict[str, float]]:
    train, test = split_data(df.select("pcaFeatures", "attack_cat_index"), config)
    label_col = "attack_cat_index"
    models_dir = config.paths.get("models_dir")
    models = {
        "naive_bayes": NaiveBayes(featuresCol="pcaFeatures", labelCol=label_col, modelType="gaussian"),
        "random_forest": RandomForestClassifier(featuresCol="pcaFeatures", labelCol=label_col),
        "decision_tree": DecisionTreeClassifier(featuresCol="pcaFeatures", labelCol=label_col),
    }

    results = {}
    for name, estimator in models.items():
        fitted = estimator.fit(train)
        predictions = fitted.transform(test)
        results[name] = evaluate_multiclass(predictions, label_col=label_col)
        if models_dir:
            fitted.write().overwrite().save(f"{models_dir}/multiclass/{name}")
    return results
