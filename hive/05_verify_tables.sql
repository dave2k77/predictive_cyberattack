USE unsw;

SELECT 'unsw_network_data_raw' AS table_name, COUNT(*) AS records
FROM unsw_network_data_raw;

SELECT 'unsw_network_data_cleaned' AS table_name, COUNT(*) AS records
FROM unsw_network_data_cleaned;

SELECT 'categorical_features' AS table_name, COUNT(*) AS records
FROM categorical_features;

SELECT 'discrete_features' AS table_name, COUNT(*) AS records
FROM discrete_features;

SELECT 'continuous_features' AS table_name, COUNT(*) AS records
FROM continuous_features;

SELECT 'model_input_features' AS table_name, COUNT(*) AS records
FROM model_input_features;
