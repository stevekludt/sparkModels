import happybase
import struct
import os
import sys
import numpy
from dateutil import parser
import pandas as pd

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.ml.regression import LinearRegression
from pyspark.mllib.util import MLUtils
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.feature import  StandardScaler
from pyspark.mllib.evaluation import RegressionMetrics


os.environ['PYSPARK_PYTHON'] = sys.executable

# Spark config
conf=SparkConf()
conf.setMaster("spark://skubuntudev:7077")
conf.setAppName("PyBuildModels")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# happybase config
connection = happybase.Connection('127.0.0.1')
table = connection.table('sensor')
connection.open()

dictList = []
for key, data in table.scan(row_prefix='A1234'):
    eventdt = key[-19:]
    sensorName = key[:5] # this needs to be updated to do a substring since sensor names can be of different length
    UTCTime = parser.parse(eventdt)
    rowDict = {'datetime': eventdt, 'sensorName': sensorName, 'UTCTime': UTCTime}

    for key1, value in data.iteritems():
        rowDict[key1] = struct.unpack(">d", value)[0]

    dictList.append(rowDict)

df = sqlContext.createDataFrame(dictList)
df.show()
pdf = df.toPandas

table = pd.pivot_table(pdf, index=['datetime'], columns=['data:temp'], aggfunc=numpy.mean)
print table.values
# For Testing
#df.show()
#df.describe(['data:temp', 'datetime', 'sensorName', 'data:humidity']).show()
df = df.select('data:temp', 'data:humidity', 'data:chlPPM', 'data:co2', 'data:flo', 'data:psi')
#df.show()
temp = df.map(lambda line:LabeledPoint(line[0], [line[1:]]))

# Scale the data
features = df.map(lambda row: row[1:])
standardizer = StandardScaler()
model = standardizer.fit(features)
features_transform = model.transform(features)
print features_transform.take(5)

lab = df.map(lambda row: row[0])

transformedData = lab.zip(features_transform)

transformedData = transformedData.map(lambda row: LabeledPoint(row[0], [row[1]]))

trainingData, testingData = transformedData.randomSplit([.8, .2], seed=1234)

lr = LinearRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

linearModel = LinearRegressionWithSGD.train(trainingData, 1000, .0002)
print linearModel.weights

print testingData.take(10)

print linearModel.predict([5.20814108601,42.4568179338,0.443700296128,6.20889144381,58.6223297308]) #actual 54.022

#score the model of the training data
prediObserRddIn = trainingData.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metrics = RegressionMetrics(prediObserRddIn)
print metrics.r2
print metrics.rootMeanSquaredError

#predict on the testing data
prediObserRddOut = testingData.map(lambda row: (float(linearModel.predict(row.features[0])), row.label))
metricsOut = RegressionMetrics(prediObserRddOut)
print metricsOut.r2
print metricsOut.rootMeanSquaredError