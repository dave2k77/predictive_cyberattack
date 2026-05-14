PYTHON ?= python3
CONFIG ?= configs/demo.yaml
HIVE_SQL ?= $(PYTHON) scripts/run_hive_sql.py --file
HIVE_WAREHOUSE_DIR ?= hdfs://localhost:9000/user/hive/warehouse
JAVA_HOME ?= $(shell dirname "$$(dirname "$$(readlink -f "$$(which java)")")")
HADOOP_HOME ?= .tools/hadoop
HADOOP_CONF_DIR ?= .runtime/hadoop-conf
HDFS ?= $(HADOOP_HOME)/bin/hdfs
HDFS_RAW_DIR ?= hdfs://localhost:9000/data/unsw/raw
HADOOP_ENV = JAVA_HOME=$(JAVA_HOME) HADOOP_HOME=$(abspath $(HADOOP_HOME)) HADOOP_CONF_DIR=$(abspath $(HADOOP_CONF_DIR))

.PHONY: install check-env check-hdfs-env check-hadoop-env hdfs-config hdfs-format hdfs-start hdfs-stop hdfs-status hdfs-wait demo score prepare train train-binary train-multiclass data-dry-run data-download data-verify hdfs-upload hive-create hive-profile hive-clean hive-features hive-verify hadoop-run validate test

install:
	$(PYTHON) -m pip install -e ".[dev]"

check-env:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.check_environment

check-hdfs-env:
	@$(HADOOP_ENV) $(HDFS) dfs -help >/dev/null
	@echo "HDFS CLI is available via $(HDFS)"

check-hadoop-env: check-hdfs-env
	@command -v beeline >/dev/null || echo "beeline is not on PATH; using Spark SQL Hive support for local SQL execution"
	@echo "Hadoop CLI is available; Hive SQL is runnable via $(HIVE_SQL)"

demo: check-env prepare train

score:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.run_scoring_pipeline --config $(CONFIG)

prepare:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.run_feature_pipeline --config $(CONFIG)

train:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.run_training_pipeline --config $(CONFIG) --task all

train-binary:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.run_training_pipeline --config $(CONFIG) --task binary

train-multiclass:
	PYTHONPATH=src $(PYTHON) -m predictive_cyberattack.pipelines.run_training_pipeline --config $(CONFIG) --task multiclass

data-dry-run:
	$(PYTHON) scripts/download_dataset.py --manifest configs/unsw_nb15_zenodo.yaml --dest data/raw/unsw_nb15 --dry-run

data-download:
	$(PYTHON) scripts/download_dataset.py --manifest configs/unsw_nb15_zenodo.yaml --dest data/raw/unsw_nb15

data-verify:
	$(PYTHON) scripts/download_dataset.py --manifest configs/unsw_nb15_zenodo.yaml --dest data/raw/unsw_nb15

hdfs-upload: check-hdfs-env hdfs-wait
	$(HADOOP_ENV) $(HDFS) dfs -mkdir -p $(HDFS_RAW_DIR)
	$(HADOOP_ENV) $(HDFS) dfs -put -f data/raw/unsw_nb15/*.csv $(HDFS_RAW_DIR)/

hive-create: hdfs-wait
	$(HADOOP_ENV) $(HIVE_SQL) hive/01_create_external_tables.sql --warehouse-dir $(HIVE_WAREHOUSE_DIR)

hive-profile: hdfs-wait
	$(HADOOP_ENV) $(HIVE_SQL) hive/02_profile_data.sql --warehouse-dir $(HIVE_WAREHOUSE_DIR)

hive-clean: hdfs-wait
	$(HADOOP_ENV) $(HIVE_SQL) hive/03_clean_transform.sql --warehouse-dir $(HIVE_WAREHOUSE_DIR)

hive-features: hdfs-wait
	$(HADOOP_ENV) $(HIVE_SQL) hive/04_export_feature_tables.sql --warehouse-dir $(HIVE_WAREHOUSE_DIR)

hive-verify: hdfs-wait
	$(HADOOP_ENV) $(HIVE_SQL) hive/05_verify_tables.sql --warehouse-dir $(HIVE_WAREHOUSE_DIR)

hadoop-run: data-verify hdfs-upload hive-create hive-profile hive-clean hive-features hive-verify
	$(MAKE) prepare CONFIG=configs/hadoop_local.yaml
	$(MAKE) train CONFIG=configs/hadoop_local.yaml
	$(MAKE) score CONFIG=configs/hadoop_local.yaml

validate:
	PYTHONPATH=src $(PYTHON) -m compileall -q src

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
hdfs-config:
	$(HADOOP_ENV) $(PYTHON) scripts/setup_hadoop_config.py --conf-dir $(HADOOP_CONF_DIR) --runtime-dir .runtime/hadoop-data --java-home $(JAVA_HOME)

hdfs-format: hdfs-config
	$(HADOOP_ENV) $(HDFS) namenode -format -force -nonInteractive

hdfs-start:
	$(HADOOP_ENV) $(HDFS) --daemon start namenode
	$(HADOOP_ENV) $(HDFS) --daemon start datanode

hdfs-stop:
	-$(HADOOP_ENV) $(HDFS) --daemon stop datanode
	-$(HADOOP_ENV) $(HDFS) --daemon stop namenode

hdfs-status:
	$(HADOOP_ENV) $(HDFS) dfsadmin -report

hdfs-wait:
	$(HADOOP_ENV) $(HDFS) dfsadmin -safemode wait
