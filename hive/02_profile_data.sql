USE unsw;

SELECT COUNT(*) AS total_records
FROM unsw_network_data_raw;

SELECT label, COUNT(*) AS records
FROM unsw_network_data_raw
GROUP BY label
ORDER BY label;

SELECT
  CASE WHEN length(trim(coalesce(attack_cat, ''))) = 0 THEN 'blank' ELSE trim(attack_cat) END AS attack_cat,
  label,
  COUNT(*) AS records
FROM unsw_network_data_raw
GROUP BY CASE WHEN length(trim(coalesce(attack_cat, ''))) = 0 THEN 'blank' ELSE trim(attack_cat) END, label
ORDER BY records DESC;

SELECT service, COUNT(*) AS records
FROM unsw_network_data_raw
GROUP BY service
ORDER BY records DESC;

SELECT COUNT(DISTINCT srcip) AS srcip_values
FROM unsw_network_data_raw;

SELECT COUNT(DISTINCT dstip) AS dstip_values
FROM unsw_network_data_raw;

SELECT COUNT(DISTINCT proto) AS proto_values
FROM unsw_network_data_raw;

SELECT COUNT(DISTINCT state) AS state_values
FROM unsw_network_data_raw;

SELECT COUNT(DISTINCT service) AS service_values
FROM unsw_network_data_raw;
