# UNSW-NB15 Data Acquisition

## Source Strategy

Use two source references:

- UNSW Research page as the authoritative dataset description, citation, and usage-rights reference.
- Zenodo DOI mirror for automated CSV downloads with stable file URLs and MD5 checksums.

The UNSW page states that the full CSV corpus contains four files:

```text
UNSW-NB15_1.csv
UNSW-NB15_2.csv
UNSW-NB15_3.csv
UNSW-NB15_4.csv
```

The same page states that the dataset contains 2,540,044 records and that academic/public use is granted, while commercial use requires agreement from the authors.

## Download

Preview the planned download:

```bash
make data-dry-run
```

Download the full CSV shards:

```bash
make data-download
```

The files are written to:

```text
data/raw/unsw_nb15/
```

The downloader verifies each file against the MD5 checksum from the Zenodo record.

To re-verify already downloaded files without re-downloading them:

```bash
make data-verify
```

The raw CSV shards do not include a header row. Each row has 49 columns. The first file may include a UTF-8 byte-order mark on the first source IP value; the Hive cleaning stage strips it.

## Stage to HDFS

After download:

```bash
make hdfs-config
make hdfs-start
make hdfs-wait
make check-hdfs-env
make hdfs-upload
```

This uploads raw CSV files to:

```text
/data/unsw/raw/
```

The Hive external table in `hive/01_create_external_tables.sql` reads from that HDFS location.

By default the Makefile uses the project-local Hadoop distribution:

```text
.tools/hadoop/bin/hdfs
```

If your Hadoop distribution already provides `hdfs`, use:

```bash
make hdfs-upload HADOOP_HOME=/path/to/hadoop HDFS=/path/to/hadoop/bin/hdfs
```

## Citation

Cite the dataset papers listed on the UNSW Research page, especially:

Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)." Military Communications and Information Systems Conference (MilCIS), 2015.
