USE unsw;

DROP TABLE IF EXISTS unsw_network_data_cleaned;

CREATE TABLE unsw_network_data_cleaned
STORED AS ORC
TBLPROPERTIES ("orc.compress" = "SNAPPY")
AS
SELECT
  regexp_replace(srcip, '^\\ufeff', '') AS srcip,
  sport,
  dstip,
  dsport,
  proto,
  state,
  dur,
  sbytes,
  dbytes,
  sttl,
  dttl,
  sloss,
  dloss,
  CASE
    WHEN service = '-' THEN 'unused'
    WHEN length(trim(coalesce(service, ''))) = 0 THEN 'unknown'
    ELSE trim(service)
  END AS service,
  CASE WHEN sttl > 0 THEN sbytes / sttl ELSE 0.0 END AS strate,
  CASE WHEN dttl > 0 THEN dbytes / dttl ELSE 0.0 END AS dtrate,
  sload,
  dload,
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
  sjit,
  djit,
  stime,
  ltime,
  sintpkt,
  dintpkt,
  tcprtt,
  synack,
  ackdat,
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
  CASE
    WHEN length(trim(coalesce(attack_cat, ''))) = 0 THEN 'None'
    WHEN trim(attack_cat) = 'Backdoors' THEN 'Backdoor'
    ELSE trim(attack_cat)
  END AS attack_cat,
  label
FROM unsw_network_data_raw
WHERE srcip <> '127.0.0.1'
  AND dstip <> '127.0.0.1';
