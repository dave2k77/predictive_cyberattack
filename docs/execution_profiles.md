# Execution Profiles

Runtime behavior is configured through YAML files in `configs/`.

## Demo Profile

`configs/demo.yaml` uses:

- Spark local mode
- `data/sample/unsw_sample.csv`
- local `data/processed/`
- local `models/`
- local `reports/metrics/`

Run:

```bash
make demo
```

This is a smoke test only. It verifies that feature preparation, PCA, training, model persistence, and metric writing work.

The demo profile writes the fitted preprocessing pipeline to:

```text
models/preprocessing_pipeline
```

## Hadoop Local Profile

`configs/hadoop_local.yaml` uses:

- Hive support
- Hive table input
- HDFS model and feature paths
- the same PySpark feature and training modules

Run:

```bash
make hdfs-upload
make hdfs-wait
make hive-create
make hive-profile
make hive-clean
make hive-features
make hive-verify
make prepare CONFIG=configs/hadoop_local.yaml
make train CONFIG=configs/hadoop_local.yaml
make score CONFIG=configs/hadoop_local.yaml
```

This is the primary project profile for demonstrating the Hadoop/Hive/PySpark stack.

The Hadoop profile writes the fitted preprocessing pipeline to:

```text
hdfs://localhost:9000/models/unsw/preprocessing_pipeline
```

Scored predictions are written to:

```text
hdfs://localhost:9000/data/unsw/predictions/latest
```

## Adding New Profiles

Add another YAML file under `configs/` and keep the same top-level sections:

```yaml
app_name: predictive-cyberattack
mode: hadoop
master: null
enable_hive: true
paths: {}
features: {}
training: {}
```

Profiles should change infrastructure and paths, not model code.
