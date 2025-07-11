##############################################################################
# PySpark Advanced Analytics: Analytics for Feature Selection                #
# by Davian Ricardo Chin                                                     #
##############################################################################

# =====================================================================================
# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
# =====================================================================================

# import SparkSession
# =====================================================
from pyspark.ml.feature import VectorAssembler
import pandas as pd
from pyspark.mllib.stat import Statistics
from pyspark.ml.stat import ChiSquareTest
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.feature import PCA

from pyspark.sql import SparkSession

# Set up a spark session
spark = SparkSession.builder.getOrCreate()

# =====================================================================================
# CATEGORICAL FEATURES OF THE UNSW-NB15 DATASET
# =====================================================================================

categorical_data = spark.read.parquet(
    "hdfs://localhost:9000/tmp/exported/CATEGORICAL/")


# Index categorical features using StandardScaler
# =================================================
categorical_indexer = StringIndexer(inputCols=['srcip', 'dstip', 'proto', 'state', 'service', 'attack_cat'], outputCols=[
                                    'srcip_index', 'dstip_index', 'proto_index', 'state_index', 'service_index', 'attack_cat_index'])
indexed_categorical_data = categorical_indexer.fit(
    categorical_data).transform(categorical_data)

# Create categorical features vector using VectorAssembler
# ==========================================================
categorical_assember = VectorAssembler(inputCols=['srcip_index', 'dstip_index', 'proto_index',
                                                  'state_index', 'service_index', 'attack_cat_index'], outputCol='categorical_features')
categorical_data_transformed = categorical_assember.transform(
    indexed_categorical_data).select('categorical_features', 'attack_cat_index')

# Pearson's ChiSquare Test for Independence
# ===========================================

# Applying the ChiSquareTest to the transformed categorical dataset
ChiSqResult = ChiSquareTest.test(
    categorical_data_transformed, "categorical_features", "attack_cat_index")

# Storing the test results into variables
degreesOfFreedom = ChiSqResult.select("degreesOfFreedom").collect()[0]
p_values = ChiSqResult.select("pValues").collect()[0]
testStatistics = ChiSqResult.select("statistics").collect()[0]

# Printing out the test results
# ===============================
print()
print("Pearson's ChiSquare Test of Independence Results")
print("=======================================================================")
print("Degrees of Freedom: {}".format(degreesOfFreedom))
print("=======================================================================")
print("P-Values: {}".format(p_values))
print("=======================================================================")
print("Test Statistics: {}".format(testStatistics))
print("=======================================================================")

categorical_features_final = categorical_data.select(
    "srcip", "dstip", "proto", "state", "service", "attack_cat")

# Checking the  number of distinct categories in categorical data (drop those with number greater than 32)
srcip_distinct = categorical_features_final.select(
    "srcip").distinct().count()  # num = 43 > 32 : drop
dstip_distinct = categorical_features_final.select(
    "dstip").distinct().count()  # num = 47 > 32 : drop
proto_distinct = categorical_features_final.select(
    "proto").distinct().count()  # num = 134 > 32 : drop
state_distinct = categorical_features_final.select(
    "state").distinct().count()  # num = 16 < 32 : keep
service_distinct = categorical_features_final.select(
    "service").distinct().count()  # num = 2 < 32 : keep
categorical_features_final.select(
    "attack_cat").distinct().count()  # label : keep

# Results of Distinct Counts
# =============================
print()
print("Distinct Category Counts for Categorical Features (num > 32: drop)")
print("======================================================================")
print("Number of distinct groupings for srcip: {}".format(srcip_distinct))
print("======================================================================")
print("Number of distinct groupings for dstip: {}".format(dstip_distinct))
print("======================================================================")
print("Number of distinct groupings for proto: {}".format(proto_distinct))
print("======================================================================")
print("Number of distinct groupings for state: {}".format(state_distinct))
print("======================================================================")
print("Number of distinct groupings for service: {}".format(service_distinct))
print("======================================================================")

# =====================================================================================
# DISCRETE FEATURES OF THE UNSW-NB15 DATASET
# =====================================================================================

discrete_data = spark.read.parquet(
    "hdfs://localhost:9000/tmp/exported/DISCRETE/")
discrete_cols = discrete_data.columns
# Assemble features into a column of features vectors
discrete_assember = VectorAssembler(
    inputCols=discrete_cols, outputCol="discreteFeatures")

discrete_data_vectors = discrete_assember.transform(
    discrete_data).select("discreteFeatures", 'label')

discrete_data_vectors_rdd = discrete_data_vectors.rdd

# Correlation Analysis for Discrete Features
# =============================================
# Computing the correlation matrix
# ===================================
discrete_cols = discrete_data.columns
discrete_rdd = discrete_data.rdd.map(lambda row: row[0:])

# Correlation Matrix
discrete_features_summary = Statistics.colStats(discrete_rdd)
discrete_corr_mat = Statistics.corr(discrete_rdd, method="pearson")
discrete_corr_mat_df = pd.DataFrame(discrete_corr_mat)
discrete_corr_mat_df.index, discrete_corr_mat_df.columns = discrete_cols, discrete_cols

# Loading the discrete correlation dataset
# =============================================
discrete_col_corr = spark.read.csv(
    '/home/davianc/Documents/cyberattack_data/data_tables/col_corr.csv', inferSchema=True, header=True)

# Dropping irrelevant feature columns using PySpark SQL Query
# ===============================================================
discrete_col_corr.createOrReplaceTempView("DATA")
result = spark.sql(
    "SELECT name, corr FROM DATA WHERE (corr >= 0.15 OR corr <= -0.15) AND name != 'ct_state_ttl'")
discrete_cols_final = result


# =====================================================================================
# CONTINUOUS FEATURES OF THE UNSW-NB15 DATASET
# =====================================================================================

# Analyse the continuous features of the dataset
continuous_data = spark.read.parquet(
    "hdfs://localhost:9000/tmp/exported/CONTINUOUS/")
unswData = spark.read.parquet("hdfs://localhost:9000/tmp/exported/UNSW_DATA/")

cont_assembler = VectorAssembler(inputCols=continuous_data.columns,
                                 outputCol='vectorFeatures')
continuous_vector_data = cont_assembler.transform(
    continuous_data).select('vectorFeatures')


# DESCRIPTIVE ANALYTICS AND KERNEL DENSITY ESTIMATION
# ===================================================================

# Descriptive Analytics
# =======================
continuous_data.describe().show(5)


# Kernel Density Plot: strate feature column
# =============================================
strate_data = continuous_data.select('strate')
strate_data_pd = strate_data.toPandas()

scaled_strate_pd = strate_data_pd / strate_data_pd.abs().max()
scaled_strate_pd.plot.kde(bw_method=3)

dtrate_data = continuous_data.select('dtrate')
dtrate_data_pd = dtrate_data.toPandas()

scaled_dtrate = dtrate_data_pd / dtrate_data_pd.abs().max()
scaled_dtrate.plot.kde(bw_method=3)



# FEATURE SELECTION
#===============================================================================
# Scaling and vectorisaing data
unsw_data = spark.read.parquet("hdfs://localhost:9000/tmp/exported/UNSW_DATA")

unsw_net_data = unsw_data.select("state", "dtrate", "service", "sload", "dload",
                                 "sintpkt", "dintpkt", "ct_state_ttl",
                                 "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                                 "ct_srv_src", "ct_src_dport_ltm",
                                 "ct_dst_sport_ltm", "ct_dst_src_ltm",
                                 "attack_cat", "label")

categorical_cols = ["state", "service", "attack_cat"]
indexed_cols = ["state_index", "service_index", "attack_cat_index"]

indexed_unsw_net_data = StringIndexer(inputCols=categorical_cols, outputCols=indexed_cols).fit(
    unsw_net_data).transform(unsw_net_data)

# UNDERSAMPLING DATA
def under_sampling_function(df):
    """
    This function takes a dataframe as argument and returns an
    under-sampled dataframe.

    """
    major_class = df.filter("label = 0")
    minor_class = df.filter("label = 1")
    class_ratio = int(major_class.count() / minor_class.count())

    sampled_major_class = major_class.sample(False, 1/class_ratio)
    under_sampled_df = sampled_major_class.unionAll(minor_class)
    return under_sampled_df


balanced_index_data = under_sampling_function(indexed_unsw_net_data)

# VECTORISING THE DATA
vectorised_balanced_data = VectorAssembler(inputCols=["state_index", "dtrate",
                                                      "service_index", "sload",
                                                      "dload", "sintpkt",
                                                      "dintpkt", "ct_state_ttl",
                                                      "ct_srv_dst",
                                                      "ct_dst_ltm",
                                                      "ct_src_ltm",
                                                      "ct_srv_src",
                                                      "ct_src_dport_ltm",
                                                      "ct_dst_sport_ltm",
                                                      "ct_dst_src_ltm",
                                                      "attack_cat_index", "label"],
                                           outputCol="features").setHandleInvalid("skip").transform(balanced_index_data)

unsw_net_data_final = vectorised_balanced_data.select(
    "features", "label", "attack_cat_index")

unsw_net_data_scaled = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True).fit(
    unsw_net_data_final).transform(unsw_net_data_final).select("scaledFeatures", "label", "attack_cat_index")


# PCA DIMENSIONALITY REDUCTION
col_names = unsw_net_data_scaled.columns
features_rdd = unsw_net_data_scaled.rdd.map(lambda x: x[0:]).toDF(col_names)


pca = PCA(k=5, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_reduced_unsw_data = pca.fit(features_rdd).transform(
    features_rdd).select('pcaFeatures', 'label', 'attack_cat_index')
