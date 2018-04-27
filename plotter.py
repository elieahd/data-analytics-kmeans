# install
# python -m pip install matplotlib

# imports
import sys
import numpy as np
import matplotlib.pyplot as plt

from pyspark import SparkContext

if len(sys.argv) != 2:
    print("1 additional arguments are needed :")
    print(" * file name ")
    print("Try executing the following command : spark-submit plotter.py out")
    exit(0)

# methods
def loadData(path):
    dataFromText = sc.textFile(path)
    return dataFromText.map(lambda x: customSplit(x))

def customSplit(x):
    value = x.split(",")
    return (value[0], float(value[1]), float(value[2]))

# inputs
file_name = sys.argv[1] + '.csv'  # file name to be generated

sc = SparkContext("local", "generator") # spark context

data = loadData(file_name)

Xs = data.map(lambda x : x[1]).collect()
Ys = data.map(lambda x : x[2]).collect()

plt.plot(Xs, Ys, 'bo')
plt.show()
