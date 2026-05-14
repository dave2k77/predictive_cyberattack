##############################################################################
# PySpark Machine Learning: Binary Classification                            #
# by Davian Ricardo Chin                                                     #
##############################################################################

# SETTING UP THE ENVIRONMENT FOR PYSPARK PROGRAMMING
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName(
    "PySpark Machine Learning: Binary Classification").getOrCreate()

# UNSW-NB15 DATASET: PROCEESED VERSION
unsw_bc_data = spark.read.parquet(
    "hdfs://localhost:9000/tmp/exported/PCA_DATA").select("pcaFeatures", "label")

# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier

# SPLITIING THE DATA INTO TRAINING AND TESTING DATSETS
train, test = unsw_bc_data.randomSplit([0.7, 0.3], seed=25)


# LOGISTIC REGRESSION CLASSIFIER
# ============================================================

# Initiate and train
lr = LogisticRegression(featuresCol="pcaFeatures",
                        labelCol='label')
lrModel = lr.fit(train)

# Testing
lrPrediction = lrModel.evaluate(test)
# lr_pred.predictions.show(5)

# Evaluating the LR Model
acc_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                              labelCol="label",
                                              metricName="areaUnderROC")

pr_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                             labelCol="label",
                                             metricName="areaUnderROC")

auc = acc_evaluator.evaluate(lrPrediction.predictions)
pr = pr_evaluator.evaluate(lrPrediction.predictions)
print("LR Classifier Accuracy Score: {0:2.2f}%".format(auc*100))
print("LR Classifier Precision Score: {0:2.2f}%".format(pr*100))

# Create a parameter grid for the cross validator
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).addGrid(
    lr.elasticNetParam, [0.2, 0.6, 0.8]).addGrid(lr.maxIter, [10, 20, 30]).build()

# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
cv_acc_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for LR Classifier: {0:2.2f}%".format(
    cv_acc_result*100))
cv_pr_result = pr_evaluator.evaluate(cvPrediction)
print("Cross Validation Precision Score for LR Classifier: {0:2.2f}%".format(
    cv_pr_result*100))


# ============================================================
# NAIVE BAYES CLASSIFIER
# ============================================================

# Initiate NB Model
nb = NaiveBayes(featuresCol='pcaFeatures',
                labelCol='label', modelType="gaussian")
# multinomial, bernoulli, gaussian


# train and test NB model
nbModel = nb.fit(train)
nbPrediction = nbModel.transform(test)

# create an evaluator for the Naive Bayes binary classifier
acc_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                              labelCol="label",
                                              metricName='areaUnderROC')
pr_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                             labelCol="label",
                                             metricName='areaUnderPR')

# calculating the accuracy of the NB model
acc = acc_evaluator.evaluate(nbPrediction)
print('Naive Bayes Classifier Accuracy Score: {0:2.2f}%'.format(acc*100))
# Prints out: Naive Bayes Accuracy: 0.9750322092256697
pr = pr_evaluator.evaluate(nbPrediction)
print('Naive Bayes Classifier Precision Score: {0:2.2f}%'.format(pr*100))

# create parameter grid
paramGrid = ParamGridBuilder().addGrid(nb.smoothing, [0, 1, 3, 5]).build()

# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=nb, estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(), numFolds=3)
cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
acc_cv_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for NB Classifier: {0:2.2f}%".format(
    acc_cv_result*100))
pr_cv_result = pr_evaluator.evaluate(cvPrediction)
print("Cross Validation Precision Score for NB Classifier: {0:2.2f}%".format(
    pr_cv_result*100))


# ============================================================
# GRADIENT BOOSTED TREE CLASSIFIER
# ============================================================

# Initiate and train
gbt = GBTClassifier(featuresCol='pcaFeatures', labelCol='label')

# Traing and test the GBT model
gbtModel = gbt.fit(train)
gbtPrediction = gbtModel.transform(test)

# create an evaluator for the Naive Bayes binary classifier
acc_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                              labelCol="label",
                                              metricName='areaUnderROC')

pr_evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction",
                                             labelCol="label",
                                             metricName='areaUnderPR')

# calculating the accuracy of the GBT model
acc = acc_evaluator.evaluate(gbtPrediction)
print(
    'Gradient Boosted Tree Classifier Accuracy Score: {0:2.2f}%'.format(acc*100))
pr = pr_evaluator.evaluate(gbtPrediction)
print(
    'Gradient Boosted Tree Classifier Precision Score: {0:2.2f}%'.format(pr*100))

# define parameter grid
paramGrid = ParamGridBuilder().build()
# Train the cross validator estimator to the training data
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvModel = cv.fit(train)
cvPrediction = cvModel.transform(test)
# Evaluate the cross validation model using the BinaryClassificationEvaluator
acc_cv_result = acc_evaluator.evaluate(cvPrediction)
print("Cross Validation Accuracy Score for GBT Classifier: {0:2.2f}%".format(
    acc_cv_result*100))
pr_cv_result = pr_evaluator.evaluate(cvPrediction)
print("Cross Validation Precision Score for GBT Classifier: {0:2.2f}%".format(
    pr_cv_result*100))
