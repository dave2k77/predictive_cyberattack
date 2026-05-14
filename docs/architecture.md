# Architecture

The project is organized as a big-data machine learning pipeline for network intrusion detection.

```text
UNSW-NB15 raw files
  -> HDFS raw zone
  -> Hive external table
  -> Hive profiling and cleaning
  -> Hive ORC feature tables
  -> PySpark feature engineering
  -> Spark ML PCA transformation
  -> Spark ML binary and multiclass classifiers
  -> metrics and model artifacts
```

## Responsibilities

Hive owns the data warehouse layer:

- schema-on-read ingestion over raw UNSW-NB15 files
- profiling queries for labels, attack categories, services, and cardinality
- canonical cleaning of categorical values
- feature-table materialization in ORC

PySpark owns the machine learning layer:

- local or HDFS-backed data loading
- input contract validation
- categorical indexing
- numeric vector assembly
- feature scaling
- PCA feature reduction
- binary and multiclass model training
- metrics and model artifact writing

## Design Choices

The pipeline uses configuration files instead of hardcoded paths. The same Python package can run in a tiny local demo profile or against Hive/HDFS paths.

The binary `label` is never included in model features. This is an explicit guardrail against target leakage from the original project.

The fitted preprocessing pipeline is persisted separately from classifier models. This preserves the learned categorical indexes, scaler parameters, and PCA projection used to create the training features.

The demo dataset is synthetic and only proves that the Spark path runs. Model quality must be evaluated with the real UNSW-NB15 dataset.
