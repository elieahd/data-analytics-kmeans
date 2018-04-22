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
    list1 = centroid[0][0]
    list2 = dataPoint[0][:4:]
    print("calculateDistance(" + str(list1) + ", " + str(list2))
    array1 = numpy.array(list1)
    array2 = numpy.array(list2)
    dist = numpy.sqrt(numpy.sum(array1 - array2)**2)
    print(str(dist))
    return (dataPoint[1], (centroid[1], dist))

def minDist(row):
    index = row[0]
    myList = row[1]
    print("MinDist(" + str(index) + "," + str(myList) + ")")
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

centroids = initCentroid(a, b, k)
#centroids.collect()

data = data.zipWithIndex()

# def findNearestCentroid(arg):
#     pass

# while True:
cartesianData = centroids.cartesian(data)
res = cartesianData.map(lambda (centroid, dataPoint): calculateDistance(centroid, dataPoint))

finalResult = res.groupByKey().map(lambda x: (x[0], list(x[1]))).map(lambda row: minDist(row))
finalResult.collect()


















# rdd = sc.parallelize([1, 2])
# rdd.cartesian(rdd).collect()

# (([6.8, 3.0, 5.5, 2.1], 0), ([5.1, 3.5, 1.4, 0.2, u'Iris-setosa'], 0))
# test.first()[1][0][:4:]
# test.first()[0][0]
# numpy.linalg.norm(numpy.array(x) - numpy.array(y))
# numpy.sqrt(numpy.sum((numpy.array(x) - numpy.array(y))**2))
# ========================================================================
# result with distance format
# (((cindex, centroid), (dataIndex, dataPoint)),distance)
# GroupBy/GroupByKey
