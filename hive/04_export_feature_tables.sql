USE unsw;

DROP TABLE IF EXISTS categorical_features;
DROP TABLE IF EXISTS discrete_features;
DROP TABLE IF EXISTS continuous_features;
DROP TABLE IF EXISTS model_input_features;

CREATE TABLE categorical_features
STORED AS ORC
TBLPROPERTIES ("orc.compress" = "SNAPPY")
AS
SELECT srcip, dstip, proto, state, service, attack_cat, label
FROM unsw_network_data_cleaned;

CREATE TABLE discrete_features
STORED AS ORC
TBLPROPERTIES ("orc.compress" = "SNAPPY")
AS
SELECT
  sport,
  dsport,
  sbytes,
  dbytes,
  sttl,
  dttl,
  sloss,
  dloss,
  spkts,
  dpkts,
  swin,
  dwin,
  stcpb,
  dtcpb,
  smeansz,
  dmeansz,
  trans_depth,
  res_bdy_len,
  is_sm_ips_ports,
  ct_state_ttl,
  ct_flw_http_mthd,
  is_ftp_login,
  ct_ftp_cmd,
  ct_srv_src,
  ct_srv_dst,
  ct_dst_ltm,
  ct_src_ltm,
  ct_src_dport_ltm,
  ct_dst_sport_ltm,
  ct_dst_src_ltm,
  label
FROM unsw_network_data_cleaned;

CREATE TABLE continuous_features
STORED AS ORC
TBLPROPERTIES ("orc.compress" = "SNAPPY")
AS
SELECT dur, strate, dtrate, sload, dload, sjit, djit, sintpkt, dintpkt, tcprtt, synack, ackdat, label
FROM unsw_network_data_cleaned;

CREATE TABLE model_input_features
STORED AS ORC
TBLPROPERTIES ("orc.compress" = "SNAPPY")
AS
SELECT
  state,
  dtrate,
  service,
  sload,
  dload,
  sintpkt,
  dintpkt,
  ct_state_ttl,
  ct_srv_dst,
  ct_dst_ltm,
  ct_src_ltm,
  ct_srv_src,
  ct_src_dport_ltm,
  ct_dst_sport_ltm,
  ct_dst_src_ltm,
  attack_cat,
  label
FROM unsw_network_data_cleaned;
