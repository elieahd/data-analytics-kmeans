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
