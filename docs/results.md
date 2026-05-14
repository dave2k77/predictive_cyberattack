# Results

These results were produced from the Hadoop-local profile against the full UNSW-NB15 CSV shards.

## Data Volumes

| Layer | Records |
| --- | ---: |
| Raw external table | 2,540,047 |
| Cleaned ORC table | 2,540,046 |
| Categorical features | 2,540,046 |
| Discrete features | 2,540,046 |
| Continuous features | 2,540,046 |
| Model input features | 2,540,046 |

The one-record difference between raw and cleaned data is expected from the localhost filter in `hive/03_clean_transform.sql`.

## Binary Classification

| Model | Accuracy | F1 | ROC AUC | PR AUC |
| --- | ---: | ---: | ---: | ---: |
| Gradient boosted tree | 0.9892 | 0.9892 | 0.9984 | 0.9970 |
| Logistic regression | 0.9854 | 0.9854 | 0.9913 | 0.9766 |
| Naive Bayes | 0.6677 | 0.6356 | 0.8172 | 0.8630 |

## Multiclass Classification

| Model | Accuracy | F1 |
| --- | ---: | ---: |
| Random forest | 0.8918 | 0.8758 |
| Decision tree | 0.8827 | 0.8715 |
| Naive Bayes | 0.7048 | 0.7253 |

## Reproducibility

The Hadoop-local run writes:

```text
hdfs://localhost:9000/data/unsw/features/pca
hdfs://localhost:9000/models/unsw/preprocessing_pipeline
hdfs://localhost:9000/models/unsw/binary
hdfs://localhost:9000/models/unsw/multiclass
reports/metrics/binary_metrics.json
reports/metrics/multiclass_metrics.json
```

The metrics are suitable as a runnable project benchmark. They should not be presented as a definitive research comparison without a stricter train/test protocol and additional leakage analysis.
