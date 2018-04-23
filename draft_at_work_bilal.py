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
