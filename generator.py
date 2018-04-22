# spark-submit generator.py out 9 3 2 10
# imports
import sys
import random

from pyspark import SparkContext
from pyspark.mllib.random import RandomRDDs

# methods

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
points = int(sys.argv[2]) # number of points to be generated --> rows
clusters = int(sys.argv[3]) # number of clusters
dimension = int(sys.argv[4]) # dimension of the data --> colones
std = int(sys.argv[5]) # standard deviation

# constants
MIN_VALUE = 0
MAX_VALUE = 100
MU = 50

sc = SparkContext("local", "generator")
normalValuesRDD = RandomRDDs.normalVectorRDD(sc, numRows = points, numCols = dimension)
(MAX_VALUE + 1 - MIN_VALUE) + MIN_VALUE
normalValuesRDD.map(lambda value : std * value)
