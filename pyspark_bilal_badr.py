from pyspark.mllib.random import RandomRDDs
import numpy
import datetime

# path = "data/iris_small.dat"
path = "data/iris_clustering.dat"
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
        #element = ([5.4, 3.7, 1.5, 0.2, u'Iris-setosa'], 0.6999999999999994)
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

data = loadData(path)
centroids = initCentroids(data, numClusters)
centroids.collect()

iterations = 0
startTime = datetime.datetime.now()
while True:
    iterations += 1
    cartesianData = centroids.cartesian(data)
    res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))
    finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
    dataByCluster = finalResult.join(data).map(lambda (iPoint, ((iCentroid, dist), data)): (iCentroid, (data, dist)))
    dataByCluster = dataByCluster.groupByKey().map(lambda (key, resultIterator): (key, list(resultIterator)))
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
print("Number of iterations until convergence: " + str(iterations))
print("Elapsed time: " + str(endTime - startTime))
