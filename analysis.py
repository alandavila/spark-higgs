# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 17:45:55 2018

@author: davil
"""

import pyspark
import os

#Data

from pyspark.sql import SparkSession

spark = SparkSession.builder.master('local').appName('higgs-analysis').getOrCreate()

data_location = os.path.join('resources','HIGGS_subsampled_20k.csv')
df = spark.read.load(data_location, format="csv", sep=",", inferSchema="true", header="true")
#df.head(2)

(training, test) = df.randomSplit([0.7, 0.3])
print(training.count(), test.count())

training.describe(training.columns[1]).show()

from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector

training_dense = training.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
training_dense = spark.createDataFrame(training_dense, ["label", "features"])

test_dense = test.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
test_dense = spark.createDataFrame(test_dense, ["label", "features"])

from pyspark.ml.feature import StandardScaler
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled", withMean=True)

scaler = standardScaler.fit(training_dense)
scaled_training = scaler.transform(training_dense)
#scaled_training.take(2)

scaled_test = scaler.transform(test_dense)
#scaled_test.take(2)

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib import linalg as mllib_linalg
from pyspark.ml import linalg as ml_linalg

def as_old(v):
    if isinstance(v, ml_linalg.SparseVector):
        return mllib_linalg.SparseVector(v.size, v.indices, v.values)
    if isinstance(v, ml_linalg.DenseVector):
        return mllib_linalg.DenseVector(v.values)
    raise ValueError("Unsupported type {0}".format(type(v)))
    
scaled_labelPoint_train = scaled_training.rdd.map(lambda row: LabeledPoint(row.label, as_old(row.features)))
#scaled_labelPoint_train.take(2)

import time
train_start = time.time()
model = GradientBoostedTrees.trainClassifier(scaled_labelPoint_train,
                                             categoricalFeaturesInfo={}, numIterations=10)
train_end = time.time()
print(f'Time elapsed training model: {train_end - train_start} seconds')

# Evaluate model on test instances and compute test error
predictions = model.predict(scaled_test.rdd.map(lambda x: x.features.values))
labelsAndPredictions = scaled_test.rdd.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(
    lambda lp: lp[0] != lp[1]).count() / float(scaled_test.rdd.count())
print('Test Error = ' + str(testErr))
print('Learned classification GBT model:')
print(model.toDebugString())

spark.stop()




















