# Predictive Cyberattack Detection

This project demonstrates an end-to-end big-data machine learning workflow for network intrusion detection using Hadoop HDFS, Hive, and PySpark ML.

The current project is the structured pipeline under `src/`, `hive/`, and `configs/`. Older root-level scripts are historical source material only and are not part of the runnable architecture.

## Architecture

```text
Raw UNSW-NB15 files
  -> HDFS raw zone
  -> Hive external table
  -> Hive profiling, cleaning, and feature tables
  -> PySpark feature engineering and PCA
  -> PySpark ML binary and multiclass classifiers
  -> metrics and model artifacts
```

The Hadoop/Hive/Spark route is the primary project path. A small synthetic demo profile is included only so the PySpark pipeline can be smoke-tested without distributing the full UNSW-NB15 dataset.

## Repository Layout

```text
configs/                      Runtime profiles for demo and Hadoop-local execution
docs/                         Architecture and stack workflow notes
hive/                         Staged Hive SQL pipeline
scripts/run_hive_sql.py       Spark SQL runner for Hive-compatible SQL files
src/predictive_cyberattack/   Config, Spark session, feature, training, and evaluation code
tests/                        Lightweight tests for config, environment parsing, and feature safeguards
data/sample/                  Tiny synthetic smoke-test data
reports/metrics/              Generated metric JSON files
models/                       Local demo model output location
legacy/                       Original coursework artifacts retained for provenance
```

## Documentation

- [Architecture](docs/architecture.md)
- [Hadoop, Hive, and Spark workflow](docs/hadoop_hive_spark.md)
- [Execution profiles](docs/execution_profiles.md)
- [UNSW-NB15 data acquisition](docs/data/unsw_nb15.md)
- [Results](docs/results.md)

## Setup

Install the package in editable mode:

```bash
make install
```

For environments where you do not want to install the package, the Makefile also sets `PYTHONPATH=src` for pipeline commands.

Spark requires Java. Verify the local prerequisites with:

```bash
make check-env
```

This project targets OpenJDK 17 for local Spark execution. Java versions newer than the Spark/Hadoop compatibility window, such as Java 25, can start the JVM but fail when Hadoop filesystem code initializes.

## Demo Smoke Test

Run the local Spark demo profile:

```bash
make demo
make score
```

This reads `data/sample/unsw_sample.csv`, builds PCA features into `data/processed/pca`, trains binary and multiclass PySpark classifiers, and writes metrics to `reports/metrics/`.
The fitted preprocessing pipeline and demo models are written under `models/`. The scoring step reloads those artifacts and writes predictions to `reports/scored/demo_predictions`.

## Hadoop/Hive/PySpark Path

Download the full UNSW-NB15 CSV shards:

```bash
make data-dry-run
make data-download
```

Then upload them to HDFS:

```bash
make hdfs-config
make hdfs-format
make hdfs-start
make hdfs-wait
make check-hdfs-env
make hdfs-upload
```

The Makefile defaults to a project-local Hadoop distribution at `.tools/hadoop`. Override `HADOOP_HOME` or `HDFS` if you have a system Hadoop installation.

Create and profile the Hive raw table:

```bash
make hive-create
make hive-profile
```

Clean and materialize feature tables:

```bash
make hive-clean
make hive-features
make hive-verify
```

Prepare PCA features and train models using the Hadoop profile:

```bash
make prepare CONFIG=configs/hadoop_local.yaml
make train CONFIG=configs/hadoop_local.yaml
make score CONFIG=configs/hadoop_local.yaml
```

For a one-command local stack run after Hadoop has been installed under `.tools/hadoop`, use:

```bash
make hdfs-start
make hadoop-run
make hdfs-stop
```

The helper script `scripts/bootstrap_local_stack.sh` runs the full local workflow. Set `FORMAT_HDFS=1` only when you intentionally want to initialize a fresh local HDFS namespace.

## Current Improvements Over the Original Scripts

- Runtime paths are config-driven instead of hardcoded.
- Hive SQL is split into clear pipeline stages and can run through Spark SQL with Hive support for local reproducibility.
- The `service` cleaning preserves real service values instead of collapsing all non-`-` values.
- The binary `label` is excluded from the feature vector to avoid target leakage.
- Binary ROC/PR metrics use `rawPrediction` rather than hard class predictions.
- PCA output is explicitly persisted before training.
- The fitted preprocessing pipeline is persisted for repeatable feature generation.
- Feature preparation validates expected columns, binary labels, and numeric castability.
- Training writes reproducible JSON metric artifacts.
- A scoring pipeline reloads persisted preprocessing and model artifacts to generate predictions.

## Notes

The demo CSV is synthetic and only proves that the code path runs. It is not a substitute for the UNSW-NB15 dataset and should not be used to evaluate model quality.

## Quality Gates

Run syntax validation and tests:

```bash
make validate
make test
```

These checks are intentionally lightweight. They validate package importability, config loading, Java-version parsing, and the guardrail that prevents the binary label from being configured as an input feature.

## Modernization Status

The modernized project is intentionally moving away from notebook-style scripts toward a reproducible big-data pipeline:

- Hive owns raw schema-on-read ingestion, profiling, cleaning, and feature table creation.
- PySpark owns distributed feature preparation, PCA, model training, and evaluation.
- Runtime behavior is config-driven so the same code can run in demo mode or Hadoop-local mode.
- Generated artifacts are written to ignored data, model, and report directories.
- Original coursework files are archived under `legacy/` and are no longer part of the active pipeline.
