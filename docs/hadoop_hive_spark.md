# Hadoop, Hive, and Spark Workflow

## HDFS Layout

Recommended local Hadoop paths:

```text
/data/unsw/raw/
/data/unsw/features/model_input
/data/unsw/features/pca
/models/unsw/preprocessing_pipeline
/models/unsw/binary
/models/unsw/multiclass
```

Raw data starts in `/data/unsw/raw/`. Hive reads this location through an external table.

The Makefile defaults to a project-local Hadoop distribution at `.tools/hadoop`. If you have a full Hadoop install elsewhere, override the Makefile variables:

```bash
make hdfs-upload HADOOP_HOME=/path/to/hadoop HDFS=/path/to/hadoop/bin/hdfs
```

The generated local config sets `fs.defaultFS` to `hdfs://localhost:9000`. Override with your own Hadoop config directory when needed:

```bash
HADOOP_CONF_DIR=/path/to/hadoop/etc/hadoop make hdfs-upload
```

For the project-local pseudo-distributed setup:

```bash
make hdfs-config
make hdfs-format
make hdfs-start
make hdfs-wait
make hdfs-status
```

## Hive Stages

The Hive scripts are intentionally split by lifecycle stage:

```text
hive/01_create_external_tables.sql
hive/02_profile_data.sql
hive/03_clean_transform.sql
hive/04_export_feature_tables.sql
hive/05_verify_tables.sql
```

Run them through the Makefile. Locally, these targets use `scripts/run_hive_sql.py`, which executes the Hive-compatible SQL through Spark SQL with Hive support and stores managed tables in the HDFS warehouse at `hdfs://localhost:9000/user/hive/warehouse`.

```bash
make hive-create
make hive-profile
make hive-clean
make hive-features
make hive-verify
```

This preserves the Hive warehouse layer without requiring a separate HiveServer2 process for the local demonstration. A dedicated Hive CLI or HiveServer2 deployment can still run the same staged SQL files.

## PySpark Stages

Feature preparation:

```bash
make prepare CONFIG=configs/hadoop_local.yaml
```

Training:

```bash
make train CONFIG=configs/hadoop_local.yaml
```

Scoring with persisted artifacts:

```bash
make score CONFIG=configs/hadoop_local.yaml
```

The feature preparation step writes PCA features before training. This makes the training stage repeatable and lets the model stage be rerun without rebuilding features.

It also writes a fitted preprocessing pipeline containing the categorical indexers, vector assembler, scaler, and PCA model. This pipeline is required for consistent inference-time feature generation.

The scoring stage reloads that fitted preprocessing pipeline plus the selected binary and multiclass models. By default it uses the strongest models from the verified run: gradient boosted tree for binary classification and random forest for multiclass attack-category prediction.

## Input Validation

The PySpark feature stage validates source data before fitting transformations:

- all configured categorical, numeric, and label columns must exist
- source data must not be empty
- binary labels must be `0` or `1`
- configured numeric features must cast cleanly to `double`

These checks are designed to fail early with a clear error before Spark ML estimators produce lower-level schema errors.

## Java Compatibility

Use OpenJDK 17 for this stack. Newer Java versions can fail inside Hadoop filesystem initialization even when the JVM itself starts.

Check the environment with:

```bash
make check-env
```

Check HDFS command availability with:

```bash
make check-hdfs-env
```

Check Hadoop/Hive command availability with:

```bash
make check-hadoop-env
```
