[
    (0, (0,1.1)),
    (1, (0,1.1)),
    (2, (1,1.1)),
    (3, (3,1.1)),
    (4, (2,1.1)),
    (5, (3,1.1)),
]

# (0, ([4.5, 2.3, 6.3, 5.1], (0,1.1)))
dataByCluster = finalResult.join(data).map(lambda (iPoint, (data, (iCentroid, dist))): (iCentroid, (data, dist)))

# (3, [([4.5, 2.3, 6.3, 5.1], 1.1), ([4.5, 2.3, 6.3, 5.1], 1.1)])
dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))

centroids = dataByCluster.map(lambda (iCentroid, clusterItems): recalculateCentroid(iCentroid, clusterItems))

def recalculateCentroid(iCentroid, clusterItems):
    allLists = [];
    for element in clusterItems:
        allLists.append(element[0])
    newCentroid = (iCentroid, numpy(allLists, axis = 0))
    return newCentroid

# ------------------------- New -----------------------------

# [
#     (0, (0,1.1)),
#     (1, (0,1.1)),
#     (2, (1,1.1)),
#     (3, (3,1.1)),
#     (4, (2,1.1)),
#     (5, (3,1.1)),
# ]

def recalculateCentroid(iCentroid, clusterItems):
    allLists = [];
    for element in clusterItems:
        allLists.append(element[0]);
    newCentroid = (iCentroid, numpy(allLists, axis = 0));
    return newCentroid;

# def computeIntraClusterDistance:


def hasConverged(centroids, newCentroids):
    for i = 0 To len(centroids):
        oldElement = centroid[i];
        newElement = newCentroids[i];
        if (!numpy.array_equal(oldElement, newElement)):
            return False
    return True

# The iterations count until convergence
# iterations = sc.accumulator(0)

centroids = initCentroid(a, b, k)
data = data.zipWithIndex()

while True:
    cartesianData = centroids.cartesian(data)
    res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))
    finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
    # (0, ([4.5, 2.3, 6.3, 5.1], (0,1.1)))
    dataByCluster = finalResult.join(data).map(lambda (iPoint, (data, (iCentroid, dist))): (iCentroid, (data, dist)))
    # (3, [([4.5, 2.3, 6.3, 5.1], 1.1), ([4.5, 2.3, 6.3, 5.1], 1.1)])
    dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))
    newCentroids = dataByCluster.map(lambda (iCentroid, clusterItems): recalculateCentroid(iCentroid, clusterItems))
    if hasConverged(centroids.collect(), newCentroids.collect()):
        break;
    centroids = newCentroids

# ------------------------- New -----------------------------

from pyspark.mllib.random import RandomRDDs
from math import floor
import numpy

# path = "data/iris_clustering.dat"
path = "data/iris_small.dat"

dataFromText = sc.textFile(path)

def loadData(row):
    sepalLength, sepalWidth, petalLength, petalWidth, className = row.split(',');
    return ([float(sepalLength), float(sepalWidth), float(petalLength), float(petalWidth), className]);

def initCentroid(minVal, maxVal, numOfCentroids):
    # centroidIndeces = RandomRDDs.uniformRDD(sc, numOfCentroids).map(lambda i: int(floor(minVal+(maxVal-minVal)*i))).collect()
    # centroids = data.takeSample(False, 4);
    # formattedCentroids = []
    # for centroid in centroids:
    #     element = centroid[:-1]
    #     formattedCentroids.append(element)
    # return sc.parallelize(formattedCentroids).zipWithIndex()
    return sc.parallelize([([5.7, 3.8, 1.7, 0.3], 0), ([6.2, 2.2, 4.5, 1.5], 1), ([6.7, 2.5, 5.8, 1.8], 2), ([6.3, 2.5, 5.0, 1.9], 3)]);

def assignToCluster(centroids, data):
    return centroids.cartesian(data)

def calculateDistance(centroid, dataPoint):
    list1 = centroid[0]
    list2 = dataPoint[0][:4:]
    array1 = numpy.array(list1)
    array2 = numpy.array(list2)
    dist = numpy.sqrt(numpy.sum(array1 - array2)**2)
    return (dataPoint[1], (centroid[1], dist))

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

data = dataFromText.map(lambda x: loadData(x))

a = 0
b = data.count()-1
k = 4

# centroids = initCentroid(a, b, k)
# #centroids.collect()

# data = data.zipWithIndex()


def recalculateCentroid(iCentroid, clusterItems):
    allLists = []
    for element in clusterItems:
        #element = ([5.4, 3.7, 1.5, 0.2, u'Iris-setosa'], 0.6999999999999994)
        allLists.append(element[0][:4:])
    averageArray = list(numpy.average(allLists, axis = 0))
    newCentroid = (iCentroid, averageArray)
    return newCentroid

# def computeIntraClusterDistance:


def hasConverged(centroids, newCentroids):
    for i in range(len(centroids)):
        oldElement = centroids[i][1];
        newElement = newCentroids[i][1];
        if not numpy.array_equal(oldElement, newElement):
            return False
    return True

# The iterations count until convergence
# iterations = sc.accumulator(0)

centroids = initCentroid(a, b, k)
data = data.zipWithIndex()

# while True:
cartesianData = centroids.cartesian(data)
res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))
finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
# (0, ([4.5, 2.3, 6.3, 5.1], (0,1.1)))

data = data.map(lambda (x, y): (y, x))

dataByCluster = finalResult.join(data).map(lambda (iPoint, ((iCentroid, dist), data)): (iCentroid, (data, dist)))
# (3, [([4.5, 2.3, 6.3, 5.1], 1.1), ([4.5, 2.3, 6.3, 5.1], 1.1)])
dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))
newCentroids = dataByCluster.map(lambda (iCentroid, clusterItems): recalculateCentroid(iCentroid, clusterItems))
if hasConverged(centroids.collect(), newCentroids.collect()):
    break;
centroids = newCentroids

# ------------------------- New -----------------------------

from pyspark.mllib.random import RandomRDDs
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

def calculateDistance(centroid, dataPoint):
    list1 = centroid[1]
    list2 = dataPoint[1][:4:]
    array1 = numpy.array(list1)
    array2 = numpy.array(list2)
    dist = numpy.sqrt(numpy.sum(array1 - array2)**2)
    return (dataPoint[0], (centroid[0], dist))

def hasConverged(centroids, newCentroids):
    for i in range(len(centroids)):
        oldElement = centroids[i][1];
        newElement = newCentroids[i][1];
        if not numpy.array_equal(oldElement, newElement):
            return False
    return True


def recalculateCentroid(iCentroid, clusterItems):
    allLists = []
    for element in clusterItems:
        #element = ([5.4, 3.7, 1.5, 0.2, u'Iris-setosa'], 0.6999999999999994)
        allLists.append(element[0][:4:])
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

data = loadData(path)
centroids = initCentroids(data, numClusters)
res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))
finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
dataByCluster = finalResult.join(data).map(lambda (iPoint, ((iCentroid, dist), data)): (iCentroid, (data, dist)))
dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))
newCentroids = dataByCluster.map(lambda (iCentroid, clusterItems): recalculateCentroid(iCentroid, clusterItems))
#Loop code
if hasConverged(centroids.collect(), newCentroids.collect()):
    break;
centroids = newCentroids
