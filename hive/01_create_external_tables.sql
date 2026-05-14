CREATE DATABASE IF NOT EXISTS unsw;
USE unsw;

CREATE EXTERNAL TABLE IF NOT EXISTS unsw_network_data_raw (
  srcip string,
  sport int,
  dstip string,
  dsport int,
  proto string,
  state string,
  dur double,
  sbytes int,
  dbytes int,
  sttl int,
  dttl int,
  sloss int,
  dloss int,
  service string,
  sload double,
  dload double,
  spkts int,
  dpkts int,
  swin int,
  dwin int,
  stcpb bigint,
  dtcpb bigint,
  smeansz int,
  dmeansz int,
  trans_depth int,
  res_bdy_len int,
  sjit double,
  djit double,
  stime bigint,
  ltime bigint,
  sintpkt double,
  dintpkt double,
  tcprtt double,
  synack double,
  ackdat double,
  is_sm_ips_ports int,
  ct_state_ttl int,
  ct_flw_http_mthd int,
  is_ftp_login int,
  ct_ftp_cmd int,
  ct_srv_src int,
  ct_srv_dst int,
  ct_dst_ltm int,
  ct_src_ltm int,
  ct_src_dport_ltm int,
  ct_dst_sport_ltm int,
  ct_dst_src_ltm int,
  attack_cat string,
  label int
)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/data/unsw/raw/';
