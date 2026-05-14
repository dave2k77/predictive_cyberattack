#!/usr/bin/env bash
set -euo pipefail

make install
make check-env
make data-download
make hdfs-config

if [ "${FORMAT_HDFS:-0}" = "1" ]; then
  make hdfs-format
fi

make hdfs-start
make hdfs-wait
make hadoop-run
