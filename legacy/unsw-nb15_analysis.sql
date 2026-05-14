	/************************************************************************************************
	*	HIVE BIG DATA ANALYSIS AND PRE-PROCESSING													*
	*																								*
	*	by Davian Ricardo Chin																		*
	*																								*
	************************************************************************************************/

/* CREATING A DATABASE */

CREATE DATABASE unswdatabase;
SHOW DATABASES;
USE unswdatabase;



/* CREATING THE UNSW-NB15 BIG DATA TABLE */

CREATE EXTERNAL TABLE unsw_network_data(
	-- specifiy table columns and data types
	srcip string, sport int, dstip string, dsport int, proto string, state string, dur float,
	sbytes int, dbytes int, sttl int, dttl int, sloss int, dloss int, service string, Sload float,
	Dload float, Spkts int, Dpkts int, swin int, dwin int, stcpb bigint, dtcpb bigint, smeansz int,
	dmeansz int, trans_depth int, res_bdy_len int, Sjit float, Djit float, Stime bigint, Ltime bigint, 
	Sintpkt float, Dintpkt float, tcprtt float, synack float, ackdat float, is_sm_ips_ports int,
	ct_state_ttl int, ct_flw_http_mthd int, is_ftp_login int, ct_ftp_cmd int, ct_srv_src int,
	ct_srv_dst int, ct_dst_ltm int, ct_src_ltm int, ct_src_dport_ltm int, ct_dst_sport_ltm int,
	ct_dst_src_ltm int, attack_cat string, Label int)
	COMMENT "UNSW-NB15 Master Dataset" -- give some description of the data stored in the table
	ROW FORMAT DELIMITED FIELDS TERMINATED BY "," -- specify the delimiter
	LINES TERMINATED BY "\n" -- specify line breaks
	STORED AS TEXTFILE -- specify storage format
	LOCATION "/tmp/dataset/"; -- specify location of data that will be loaded into the table



/* DESCRIPTIVE STATISTICS ON CATEGORICAL DATA */

-- display table schema and other information
DESCRIBE FORMATTED unsw_network_data;

-- display the first 5 records (rows) of the data
SELECT * FROM unsw_network_data LIMIT 5; 

-- display the total number of records (rows) in the dataset
SELECT COUNT(*) AS total_records FROM unsw_network_data; 

-- exploring the attack_cat column
SELECT attack_cat FROM unsw_network_data LIMIT 20;

-- count the number of blank entries
SELECT attack_cat AS attack_category, COUNT(*) AS instances
FROM unsw_network_data
WHERE LENGTH(attack_cat) = 0
GROUP BY attack_cat;

-- view unique attack_cat values and distribution
SELECT DISTINCT attack_cat FROM unsw_network_data;

SELECT attack_cat, COUNT(*)
FROM unsw_network_data
GROUP BY attack_cat; 

-- investigate the association between label values and attack_cat values
SELECT Label AS label, COUNT(*) AS instances
FROM unsw_network_data
WHERE LENGTH(attack_cat) = 0
GROUP BY Label;

SELECT COUNT (*) total
FROM unsw_network_data
WHERE label = 0;

-- exploring the service column
SELECT service
FROM unsw_network_data
LIMIT 10;

-- count the number of " - " values in the service column
SELECT service AS service, COUNT(*) AS instances
FROM unsw_network_data
WHERE service = "-"
GROUP BY service;

-- view first few rows of the srcip and dstip columns
SELECT srcip, dstip
FROM unsw_network_data
LIMIT 5;

-- count the number of unique values
SELECT srcip, COUNT(*)
FROM unsw_network_data
GROUP BY srcip;

SELECT dstip, COUNT(*)
FROM unsw_network_data
GROUP BY dstip;

-- the most srcip address
SELECT srcip, COUNT(*)
FROM unsw_network_data
GROUP BY srcip
ORDER BY 2 DESC
LIMIT 1;

SELECT srcip, COUNT(*)
FROM unsw_network_data
GROUP BY srcip
ORDER BY 2 DESC
LIMIT 5;

-- the most dstip address
SELECT dstip, COUNT(*)
FROM unsw_network_data
GROUP BY dstip
ORDER BY 2 DESC
LIMIT 1;

SELECT dstip, COUNT(*)
FROM unsw_network_data
GROUP BY dstip
ORDER BY 2 DESC
LIMIT 5;

--view the number unique proto, state and service values
SELECT COUNT(DISTINCT proto)
FROM unsw_network_data;

SELECT COUNT(DISTINCT service)
FROM unsw_network_data;

-- top 5 most common network protocols
SELECT proto, COUNT(*)
FROM unsw_network_data
GROUP BY proto
ORDER BY 2 ASC
LIMIT 5;

-- the most common network protocol
SELECT proto, COUNT(*)
FROM unsw_network_data
GROUP BY proto
ORDER BY 2 DESC
LIMIT 1;

-- the most common network service
SELECT service, COUNT(*)
FROM categorical_features
GROUP BY service
ORDER BY 2 DESC
LIMIT 1;

SELECT DISTINCT state, COUNT(*)
FROM unsw_network_data
GROUP BY state;

-- select the modal srcip
SELECT srcip, COUNT(*)
FROM unsw_network_data
GROUP BY srcip
ORDER BY 2 DESC
LIMIT 1;

-- top 5 most common source IP addresses
SELECT srcip, COUNT(*)
FROM unsw_network_data
GROUP BY srcip
ORDER BY 2 DESC
LIMIT 5;

-- top 5 most common destination IP addresses
SELECT dstip, COUNT(*)
FROM unsw_network_data
GROUP BY dstip
ORDER BY 2 DESC
LIMIT 5;

-- modal destination IP address
SELECT dstip, COUNT(*)
FROM unsw_network_data
GROUP BY dstip
ORDER BY 2 DESC
LIMIT 1;



/* DESCRIPTIVE STATISTICS ON DISCRETE DATA */


SELECT ROUND(AVG(sbytes), 2) AS avg_sbytes, ROUND(AVG(dbytes), 2)
AS avg_dbytes, ROUND(VARIANCE(sbytes), 2)
AS var_sbytes, ROUND(VARIANCE(dbytes), 2)
AS var_dbytes
FROM discrete_features;

SELECT
SUM(sbytes) AS total_src_bytes,
SUM(sttl) AS total_src_time,
SUM(dbytes) AS total_dst_bytes,
SUM(dttl) AS total_dst_time,
ROUND(AVG(sbytes/sttl), 2) AS src_byte_rate,
ROUND(AVG(dbytes/dttl), 2) AS dst_byte_rate
FROM discrete_features;



/* DESCRIPTIVE STATISTICS ON CONTINUOUS FEATURES */


SELECT
AVG(sintpkt) AS avg_sintpkt,
STDDEV(sintpkt) AS stddev_dintpkt,
AVG(dintpkt) AS avg_dintpkt,
STDDEV(dintpkt) AS stddev_dintpkt
FROM continuous_features;

SELECT CORR(sintpkt, sload) AS sintpkt_sload_corr
FROM continuous_features;

SELECT CORR(dintpkt, dload) AS dintpkt_dload_corr
FROM continuous_features;




-- DATA CLEANING STAGE 1
CREATE TABLE unsw_network_data_first AS SELECT srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss,
if(service =='-', regexp_replace(service, "-", "unused"), 'ftp-data') AS service, Sload, Dload, Spkts, Dpkts, swin, dwin, stcpb, dtcpb,
smeansz, dmeansz, trans_depth, res_bdy_len, Sjit, Djit, from_unixtime(Stime, "dd-MM-yyyy HH:mm:ss") AS Stime,
from_unixtime(Ltime, "dd-MM-yyyy HH:mm:ss") AS Ltime, Sintpkt, Dintpkt, tcprtt, synack, ackdat, is_sm_ips_ports,ct_state_ttl, ct_flw_http_mthd,
is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, 
trim(attack_cat) AS attack_cat, Label
FROM unsw_network_data;

-- DATA CLEANING STAGE 2
CREATE TABLE unsw_network_data_second AS SELECT srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service,
Sload, Dload, Spkts, Dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth, res_bdy_len, Sjit, Djit, Stime, Ltime, Sintpkt, Dintpkt,
tcprtt, synack, ackdat, is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm,
ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm, regexp_replace(attack_cat, 'Backdoors', 'Backdoor') AS attack_cat, Label AS label
FROM unsw_network_data_first;

-- DATA CLAENING STAGE 3
CREATE TABLE unsw_network_data_cleaned AS SELECT srcip, sport, dstip, dsport, proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss, service,
Sload, Dload, Spkts, Dpkts, swin, dwin, stcpb, dtcpb, smeansz, dmeansz, trans_depth, res_bdy_len, Sjit, Djit, Stime, Ltime, Sintpkt, Dintpkt, tcprtt,
synack, ackdat, is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm,
ct_dst_sport_ltm, ct_dst_src_ltm, if(LENGTH(attack_cat) > 0, attack_cat, "None") AS attack_cat, Label AS label
FROM unsw_network_data_second;




-- SPLITTING THE DATA SET

CREATE TABLE categorical_features STORED AS ORC tblproperties("orc.compress"="SNAPPY")
AS SELECT srcip AS scrip, dstip AS dstip, proto AS proto, state AS state, service AS service, stime AS stime,
ltime AS ltime, attack_cat AS attack_cat
FROM unsw_network_data_cleaned; 

CREATE TABLE discrete_features STORED AS ORC tblproperties("orc.compress"="SNAPPY")
AS SELECT sport AS sport,dsport AS dsport,sbytes AS sbytes,dbytes AS dbytes,sttl AS sttl,dttl AS dttl,
sloss AS sloss,dloss AS dloss,spkts AS spkts,dpkts AS dpkts,swin AS swin,dwin AS dwin,stcpb AS stcpb,
dtcpb AS dtcpb,smeansz AS smeansz,dmeansz AS dmeansz,trans_depth AS trans_depth,res_bdy_len AS res_bdy_len,
is_sm_ips_ports AS is_sm_ips_ports,ct_state_ttl AS ct_state_ttl,ct_flw_http_mthd AS ct_flw_http_mthd,
is_ftp_login AS is_ftp_login,ct_ftp_cmd AS ct_ftp_cmd,ct_srv_src AS ct_srv_src,ct_srv_dst AS ct_srv_dst,
ct_dst_ltm AS ct_dst_ltm,ct_src_ltm AS ct_src_ltm,ct_src_dport_ltm AS ct_src_dport_ltm,
ct_dst_sport_ltm AS ct_dst_sport_ltm,ct_dst_src_ltm AS ct_dst_src_ltm,label AS label
FROM unsw_network_data_cleaned;

CREATE TABLE continuous_features STORED AS ORC tblproperties("orc.compress"="SNAPPY")
AS SELECT dur AS dur, sload AS sload, dload AS dload, sjit AS sjit, djit AS djit, sintpkt AS sintpkt,
dintpkt AS dintpkt, tcprtt AS tcprtt, synack AS synack, ackdat AS ackdat
FROM unsw_network_data_cleaned;



-- EXPORTING HIVE TABLES TO HDFS
INSERT OVERWRITE DIRECTORY '/tmp/exported/attack_category_distribution/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM attack_category_distribution;
INSERT OVERWRITE DIRECTORY '/tmp/exported/label_distribution/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM label_distribution;
INSERT OVERWRITE DIRECTORY '/tmp/exported/clean_data/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM unsw_network_data_cleaned;
INSERT OVERWRITE DIRECTORY '/tmp/exported/protocol_distribution/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM protocol_distribution;
INSERT OVERWRITE DIRECTORY '/tmp/exported/duration_distribution/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM duration_distribution;
INSERT OVERWRITE DIRECTORY '/tmp/exported/transaction_data/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM transaction_data;
INSERT OVERWRITE DIRECTORY '/tmp/exported/categorical_features/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM categorical_features;
INSERT OVERWRITE DIRECTORY '/tmp/exported/discrete_features/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM discrete_features;
INSERT OVERWRITE DIRECTORY '/tmp/exported/continuous_features/' ROW FORMAT DELIMITED FIELDS TERMINATED BY "," LINES TERMINATED BY "\n" SELECT * FROM continuous_features;
