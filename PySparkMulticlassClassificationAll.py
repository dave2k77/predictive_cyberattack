##############################################################################
# PySpark Machine Learning: Multiclass Classification                        #
# by Davian Ricardo Chin
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName(
    "PySpark Machine Learning: Multiclass Classification").getOrCreate()

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, DecisionTreeClassifier, NaiveBayes

unsw_mc_data = spark.read.parquet(
    "hdfs://localhost:9000/tmp/exported/PCA_DATA").select("pcaFeatures", 'attack_cat_index')

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_mc_data.randomSplit([0.7, 0.3], seed=25)

# ============================================================
# NAIVE BAYES CLASSIFIER
# ============================================================

# Initiate NB Model
nb = NaiveBayes(featuresCol='pcaFeatures',
                labelCol='attack_cat_index', modelType="gaussian")
# multinomial, bernoulli, gaussian


# train and test NB model
nbModel = nb.fit(train)
nbPrediction = nbModel.transform(test)

# create an evaluator for the Naive Bayes binary classifier
acc_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                  labelCol="attack_cat_index",
                                                  metricName='accuracy')

f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='f1')

wp_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='weightedPrecision')

# calculating the accuracy of the NB model
acc = acc_evaluator.evaluate(nbPrediction)
print('Naive Bayes Classifier Accuracy Score: {0:2.2f}%'.format(acc*100))

F1 = f1_evaluator.evaluate(nbPrediction)
print('Naive Bayes Classifier F1 Score: {0:2.2f}%'.format(F1*100))

wP = wp_evaluator.evaluate(nbPrediction)
print(
    'Naive Bayes Classifier Weighted Precision Score: {0:2.2f}%'.format(wP*100))

# create parameter grid
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0, 1, 3, 5]).build()

# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid,
                    evaluator=MulticlassClassificationEvaluator(
                        predictionCol="prediction",
                        labelCol="attack_cat_index"), numFolds=3)

cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)

# Evaluate the cross validation model using the MulticlassClassificationEvaluator
acc_cv_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for NB Classifier: {0:2.2f}%".format(
    acc_cv_result*100))
f1_cv_result = f1_evaluator.evaluate(cvPrediction)
print("Cross Validation F1 Score for NB Classifier: {0:2.2f}%".format(
    f1_cv_result*100))
wp_cv_result = wp_evaluator.evaluate(cvPrediction)
print("Cross Validation Weighted Precision Score for NB Classifier: {0:2.2f}%".format(
    wp_cv_result*100))


# ===========================================================
# RANDOM FOREST CLASSIFIER
# ===========================================================

# Instantiate the RF classifier
rf = RandomForestClassifier(
    labelCol="attack_cat_index", featuresCol="pcaFeatures")

# Training the RF model
rfModel = rf.fit(train)

# Testing the RF model
rfPrediction = rfModel.transform(test)


# create an evaluators for the Random Forest Classifier
acc_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                  labelCol="attack_cat_index",
                                                  metricName='accuracy')

f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='f1')

wp_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='weightedPrecision')

# calculating the accuracy of the RF model
acc = acc_evaluator.evaluate(rfPrediction)
print('Random Forest Classifier Accuracy Score: {0:2.2f}%'.format(acc*100))

F1 = f1_evaluator.evaluate(rfPrediction)
print('Random Forest Classifier F1 Score: {0:2.2f}%'.format(F1*100))

wP = wp_evaluator.evaluate(rfPrediction)
print(
    'Random Forest Classifier Weighted Precision Score: {0:2.2f}%'.format(wP*100))

# Create a parameter grid for the cross validator
paramGridRF = ParamGridBuilder().addGrid(
    rf.numTrees, [5, 20, 50]).addGrid(rf.maxDepth, [2, 5, 10]).build()

cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGridRF,
                    evaluator=MulticlassClassificationEvaluator(
                        predictionCol="prediction",
                        labelCol="attack_cat_index"), numFolds=3)

# Train and test the Random Forest Classifier
cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)

# Evaluate the cross validation model using the MulticlassClassificationEvaluator
acc_cv_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for RF Classifier: {0:2.2f}%".format(
    acc_cv_result*100))

f1_cv_result = f1_evaluator.evaluate(cvPrediction)
print("Cross Validation F1 Score for RF Classifier: {0:2.2f}%".format(
    f1_cv_result*100))

wp_cv_result = wp_evaluator.evaluate(cvPrediction)
print("Cross Validation Weighted Precision Score for RF Classifier: {0:2.2f}%".format(
    wp_cv_result*100))


# ===========================================================
# DECISION TREE CLASSIFIER
# ===========================================================

# Instantiate the RF classifier
dt = DecisionTreeClassifier(
    labelCol="attack_cat_index", featuresCol="pcaFeatures")

# training the DT model
dtModel = dt.fit(train)
# testing the RF model
dtPrediction = dtModel.transform(test)

# create an evaluator for the DT Classifier
acc_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                  labelCol="attack_cat_index",
                                                  metricName='accuracy')

f1_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='f1')

wp_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",
                                                 labelCol="attack_cat_index",
                                                 metricName='weightedPrecision')

# calculating the accuracy of the DT model
acc = acc_evaluator.evaluate(dtPrediction)
print('DT Classifier Accuracy Score: {0:2.2f}%'.format(acc*100))

F1 = f1_evaluator.evaluate(dtPrediction)
print('DT Classifier F1 Score: {0:2.2f}%'.format(F1*100))

wP = wp_evaluator.evaluate(dtPrediction)
print(
    'DT Classifier Weighted Precision Score: {0:2.2f}%'.format(wP*100))

# APPLYING AND EVALUATING A CROSS VALIDATION MODEL WITH RANDOM FOREST CLASSIFIER
# Create a parameter grid for the cross validator
paramGridDT = ParamGridBuilder().addGrid(dt.maxDepth, [2, 5, 10]).build()

cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGridDT,
                    evaluator=MulticlassClassificationEvaluator(
                        predictionCol="prediction",
                        labelCol="attack_cat_index"), numFolds=3)
cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)

# Evaluate the cross validation model using the MulticlassClassificationEvaluator
acc_cv_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for DT Classifier: {0:2.2f}%".format(
    acc_cv_result*100))

f1_cv_result = f1_evaluator.evaluate(cvPrediction)
print("Cross Validation F1 Score for DT Classifier: {0:2.2f}%".format(
    f1_cv_result*100))

wp_cv_result = wp_evaluator.evaluate(cvPrediction)
print("Cross Validation Weighted Precision Score for DT Classifier: {0:2.2f}%".format(
    wp_cv_result*100))
