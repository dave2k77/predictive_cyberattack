from __future__ import annotations

from pyspark.ml import Pipeline
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.feature import PCA, StandardScaler, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from predictive_cyberattack.config import ProjectConfig
from predictive_cyberattack.spark.validation import normalize_input_schema, validate_input_dataframe


def clean_model_input(df: DataFrame) -> DataFrame:
    return (
        df.withColumn(
            "service",
            F.when(F.col("service") == "-", F.lit("unused"))
            .when(F.length(F.trim(F.coalesce(F.col("service"), F.lit("")))) == 0, F.lit("unknown"))
            .otherwise(F.trim(F.col("service"))),
        )
        .withColumn(
            "attack_cat",
            F.when(F.length(F.trim(F.coalesce(F.col("attack_cat"), F.lit("")))) == 0, F.lit("None"))
            .when(F.trim(F.col("attack_cat")) == "Backdoors", F.lit("Backdoor"))
            .otherwise(F.trim(F.col("attack_cat"))),
        )
    )


def undersample_majority_class(df: DataFrame, seed: int) -> DataFrame:
    majority = df.filter(F.col("label") == 0)
    minority = df.filter(F.col("label") == 1)
    majority_count = majority.count()
    minority_count = minority.count()

    if majority_count == 0 or minority_count == 0 or majority_count <= minority_count:
        return df

    fraction = minority_count / majority_count
    return majority.sample(withReplacement=False, fraction=fraction, seed=seed).unionByName(minority)


def build_feature_dataframe(df: DataFrame, config: ProjectConfig) -> tuple[DataFrame, PipelineModel]:
    features = config.features
    training = config.training
    categorical_cols = list(features["categorical"])
    numeric_cols = list(features["numeric"])
    seed = int(training.get("seed", 25))

    validate_input_dataframe(df, config)
    normalized = normalize_input_schema(df, config)
    cleaned = clean_model_input(normalized)
    if bool(training.get("sample_majority_class", False)):
        cleaned = undersample_majority_class(cleaned, seed)

    index_output_cols = [f"{column}_index" for column in categorical_cols]
    indexers = [
        StringIndexer(
            inputCol=input_col,
            outputCol=output_col,
            handleInvalid="keep",
        )
        for input_col, output_col in zip(categorical_cols, index_output_cols, strict=True)
    ]

    # Intentionally exclude the binary label from the feature vector to prevent target leakage.
    assembler = VectorAssembler(
        inputCols=[col for col in index_output_cols if col != "attack_cat_index"] + numeric_cols,
        outputCol="features",
        handleInvalid="skip",
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
    pca = PCA(
        k=int(training.get("pca_components", 5)),
        inputCol="scaledFeatures",
        outputCol="pcaFeatures",
    )

    pipeline = Pipeline(stages=[*indexers, assembler, scaler, pca])
    model = pipeline.fit(cleaned)

    feature_df = model.transform(cleaned).select("pcaFeatures", "label", "attack_cat_index")
    return feature_df, model
