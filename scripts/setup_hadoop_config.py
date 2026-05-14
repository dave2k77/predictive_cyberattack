from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


CORE_SITE = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>hdfs://localhost:9000</value>
  </property>
  <property>
    <name>hadoop.tmp.dir</name>
    <value>{runtime_dir}/tmp</value>
  </property>
</configuration>
"""


HDFS_SITE = """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.permissions.enabled</name>
    <value>false</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>file://{runtime_dir}/namenode</value>
  </property>
  <property>
    <name>dfs.datanode.data.dir</name>
    <value>file://{runtime_dir}/datanode</value>
  </property>
  <property>
    <name>dfs.namenode.http-address</name>
    <value>localhost:9870</value>
  </property>
</configuration>
"""


HADOOP_ENV = """export JAVA_HOME="{java_home}"
export HADOOP_LOG_DIR="{log_dir}"
export HADOOP_PID_DIR="{pid_dir}"
export HADOOP_OPTS="$HADOOP_OPTS -Djava.net.preferIPv4Stack=true"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render local pseudo-distributed Hadoop configuration.")
    parser.add_argument("--conf-dir", default=".runtime/hadoop-conf")
    parser.add_argument("--runtime-dir", default=".runtime/hadoop-data")
    parser.add_argument("--java-home", default=os.environ.get("JAVA_HOME", ""))
    args = parser.parse_args()

    if not args.java_home:
        raise SystemExit("JAVA_HOME is required.")

    project_root = Path.cwd()
    conf_dir = (project_root / args.conf_dir).resolve()
    runtime_dir = (project_root / args.runtime_dir).resolve()
    log_dir = runtime_dir / "logs"
    pid_dir = runtime_dir / "pids"

    for directory in (conf_dir, runtime_dir, log_dir, pid_dir):
        directory.mkdir(parents=True, exist_ok=True)

    (conf_dir / "core-site.xml").write_text(
        CORE_SITE.format(runtime_dir=runtime_dir),
        encoding="utf-8",
    )
    (conf_dir / "hdfs-site.xml").write_text(
        HDFS_SITE.format(runtime_dir=runtime_dir),
        encoding="utf-8",
    )
    (conf_dir / "hadoop-env.sh").write_text(
        HADOOP_ENV.format(java_home=args.java_home, log_dir=log_dir, pid_dir=pid_dir),
        encoding="utf-8",
    )
    (conf_dir / "workers").write_text("localhost\n", encoding="utf-8")

    hadoop_home = os.environ.get("HADOOP_HOME")
    if hadoop_home:
        source_log4j = Path(hadoop_home) / "etc" / "hadoop" / "log4j.properties"
        if source_log4j.exists():
            shutil.copyfile(source_log4j, conf_dir / "log4j.properties")

    print(f"Wrote Hadoop config to {conf_dir}")
    print(f"Hadoop runtime data will use {runtime_dir}")


if __name__ == "__main__":
    main()
