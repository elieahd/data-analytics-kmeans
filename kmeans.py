# spark-submit kmeans.py data/iris_small.dat 4 10
# imports
import numpy
import datetime
import sys

from pyspark.mllib.random import RandomRDDs
from pyspark import SparkContext

# methods
def customSplit(row):
    values = row[0]
    index = row[1]
    dataItems = values.split(',')
    for i in range(len(dataItems) - 1):
        dataItems[i] = float(dataItems[i])
    return (index, dataItems)

def loadData(path):
    dataFromText = sc.textFile(path)
    dataZipped = dataFromText.zipWithIndex()
    return dataZipped.map(lambda x: customSplit(x))

#Random is not efficient!
def initCentroids(data, numClusters):
    sample = sc.parallelize(data.takeSample(False, numClusters))
    centroids = sample.map(lambda point : point[1][:-1])
    return centroids.zipWithIndex().map(lambda point : (point[1], point[0]))

def calculateDistance(centroid, dataPoint):
    list1 = centroid[1]
    list2 = dataPoint[1][:-1]
    array1 = numpy.array(list1)
    array2 = numpy.array(list2)
    dist = numpy.sqrt(numpy.sum(array1 - array2)**2)
    return (dataPoint[0], (centroid[0], dist))

def minDist(row):
    index = row[0]
    myList = row[1]
    minValue = -1
    minPoint = None
    minCentroidIndex = -1
    for element in myList:
        centroidIndex = element[0]
        distance = element[1]
        if (minValue == -1) or (distance < minValue):
            minValue = distance
            minCentroidIndex = centroidIndex
            minPoint = (minCentroidIndex, minValue)
    return (index, minPoint)

def recalculateCentroid(iCentroid, clusterItems):
    allLists = []
    for element in clusterItems:
        # element = ([5.4, 3.7, 1.5, 0.2, u'Iris-setosa'], 0.6999999999999994)
        allLists.append(element[0][:-1])
    averageArray = list(numpy.average(allLists, axis = 0))
    newCentroid = (iCentroid, averageArray)
    return newCentroid

def hasConverged(centroids, newCentroids):
    for i in range(len(centroids)):
        oldElement = centroids[i][1];
        newElement = newCentroids[i][1];
        if not numpy.array_equal(oldElement, newElement):
            return False
    return True

def custom(clusterId, data):
    result = [];
    for element in data:
        r = (str(clusterId) + "," + str(element[0]).replace("[", "").replace("]","") + "," + str(element[1]))
        result.append(r)
    return result;


if len(sys.argv) != 4:
    print("3 additional arguments are needed :")
    print(" * name of the file containing the points e.g. data/iris_small.dat")
    print(" * number of clusters e.g. 4")
    print(" * max number of iterations e.g. 10\n")
    print("Try executing the following command : spark-submit kmeans.py data/iris_small.dat 4 10")
    exit(0)

# inputs
path = sys.argv[1]  # file name of the points
numClusters = int(sys.argv[2]) # number of clusters
maxIterations = int(sys.argv[3]) # maximum number of iterations

sc = SparkContext("local", "generator") # spark context

data = loadData(path)
centroids = initCentroids(data, numClusters)
centroids.collect()

iterations = 0
startTime = datetime.datetime.now()
while iterations != maxIterations:
    iterations += 1
    cartesianData = centroids.cartesian(data)
    res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))
    finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
    dataByCluster = finalResult.join(data).map(lambda (iPoint, ((iCentroid, dist), data)): (iCentroid, (data, dist)))
    dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))
    # print(dataByCluster.collect())
    # print()
    newCentroids = dataByCluster.map(lambda (iCentroid, clusterItems): recalculateCentroid(iCentroid, clusterItems))
    centroidsList = centroids.collect()
    newCentroidsList = newCentroids.collect()
    if hasConverged(centroidsList, newCentroidsList):
        break;
    # To break the lineage and make the algorithm more efficient,
    # we tell spark to create a new RDD from the newCentroidsList
    # instead of using the old one.
    centroids = sc.parallelize(newCentroidsList)

endTime = datetime.datetime.now()
centroids.collect()
print("Elapsed time: " + str(endTime - startTime))
print("Number of iterations: " + str(iterations))
print("Final distance: " + str(iterations))
# print(dataByCluster.collect())

plotData = dataByCluster.map(lambda (clusterId, data) : custom(clusterId, data)).flatMap(lambda x: x)
with open('testing.csv','wb') as file:
    for row in plotData.collect():
        file.write(row)
        file.write('\n')
