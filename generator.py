# spark-submit generator.py out 9 3 2 10
# imports
import sys
import random
import numpy

from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs

# constants
MIN_MEAN_VALUE = 0
MAX_MEAN_VALUE = 100
STEPS = 0.1

# methods
def calculate_value_point(point):
    values = point[0]
    cluster = int(point[1])
    return point

# main code
if len(sys.argv) != 6:
    print("5 arguments are needed :")
    print(" * file name to be generated e.g. output")
    print(" * number of points to be generated e.g. 9")
    print(" * number of clusters e.g. 3")
    print(" * dimension of the data e.g. 2")
    print(" * standard deviation e.g. 10\n")
    print("Try executing the following command : spark-submit generator.py out 9 3 2 10")
    exit(0)

# inputs
file_name = sys.argv[1] + '.csv' # file name to be generated
points = int(sys.argv[2]) # number of points to be generated
count_cluster = int(sys.argv[3]) # number of clusters
dimension = int(sys.argv[4]) # dimension of the data
std = int(sys.argv[5]) # standard deviation

sc = SparkContext("local", "generator") # spark context

# array of the clusters : clusters = [0, 1, 2]
clusters = sc.parallelize(range(0, count_cluster))

# random means of each cluster : means_cluster = [ (0, [0.6, 80.9]), (1, [57.8, 20.2]), (2, [15.6, 49.9]) ]
means_clusters = clusters.map(lambda cluster : (cluster, random.sample(numpy.arange(MIN_MEAN_VALUE, MAX_MEAN_VALUE, STEPS), dimension)))

# creating random vector using normalVectorRDD
random_values_vector = RandomRDDs.normalVectorRDD(sc, numRows = points, numCols = dimension, numPartitions = count_cluster, seed = 1L)

# assiging a random cluster for each point
cluster_normal_vector = random_values_vector.map(lambda point : (point.tolist(), random.randint(0, count_cluster - 1)))

# re-calculating each value of each point based on the mean of a cluster
data = cluster_normal_vector.map(lambda point : calculate_value_point(point))
