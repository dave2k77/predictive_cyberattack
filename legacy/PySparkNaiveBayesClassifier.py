from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Naive Bayes Binary Classification Model").getOrCreate()

unswdata = spark.read.parquet("hdfs://localhost:9000/tmp/exported/pca_reduced/", inferSchema=True, header=True).select("pcaFeatures", "label")

train, test =unswdata.randomSplit([0.7, 0.3], 25)

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

nbc = NaiveBayes(featuresCol='pcaFeatures', labelCol='label', modelType="gaussian")
# multinomial, bernoulli, gaussian


nbm = nbc.fit(train)

nbc_pred = nbm.transform(test)

# create an evaluator for the Naive Bayes binary classifier
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")

# calculating the accuracy of the NB model
acc = evaluator.evaluate(nbc_pred)
print('Naive Bayes Accuracy: {}'.format(acc))
# Prints out: Naive Bayes Accuracy: 0.9750322092256697

paramGrid = ParamGridBuilder().build()

# Train the cross validator estimator to the training data
nb_cv = CrossValidator(estimator=nbc, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

cvm = nb_cv.fit(train)
nb_cv_pred = cvm.transform(test)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
nb_cv_result = evaluator.evaluate(nb_cv_pred)
print()
print("Cross Validation Accuracy Score for NB Classifier: {}".format(nb_cv_result))
# Prints out: Cross Validation Accuracy Score for NB Classifier: 0.9750322092256697
