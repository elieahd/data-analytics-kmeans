# imports
from pyspark.mllib.random import RandomRDDs
from math import floor
import numpy

# path = "data/iris_clustering.dat"
path = "data/iris_small.dat"
numClusters = 4

# methods
def customSplit(row):
    values = row[0]
    index = row[1]
    sepalLength, sepalWidth, petalLength, petalWidth, cluster = values.split(',')
    return (index, [float(sepalLength), float(sepalWidth), float(petalLength), float(petalWidth), cluster])

def loadData(path):
    dataFromText = sc.textFile(path)
    dataZipped = dataFromText.zipWithIndex()
    return dataZipped.map(lambda x: customSplit(x))

def initCentroids(data, numClusters):
    sample = sc.parallelize(data.takeSample(False, numClusters))
    centroids = sample.map(lambda point : point[1][:-1]) 
    return centroids.zipWithIndex().map(lambda point : (point[1], point[0]))

data = loadData(path)
centroids = initCentroids(data, numClusters)