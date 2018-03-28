## stub for td01 ex1
## execute with spark-submit iris.py

from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import FloatType

# Create spark session
spark = SparkSession.builder.master("local").appName("iris").getOrCreate()

# build the scema to load the dataframe.
colnames = ["x1", "x2", "x3", "x4", "y" ]
schema = StructType ( [ StructField(colname, FloatType(), False) for colname in colnames ] )
# assembler group all x1..x2 into a single col called X
assembler = VectorAssembler( inputCols = colnames[:-1], outputCol="X" )

## TRAINING

# load the data into the dataframe
training = spark.read.csv('data-analytics-course/td01/data/iris_bin.train', schema = schema)
training = assembler.transform(training) #group all x1..x2 into a single col called X

# keep X and y only
training = training.select("X", "y")

print("Schema: ")
training.printSchema()

print("Data")
print(training.show())


lr = LogisticRegression(maxIter=100, regParam=0.01, elasticNetParam=0.8, featuresCol = "X", labelCol = "y")
lrModel = lr.fit(training)

print("Coefficient: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

##PREVISION

colnames = ["x1", "x2", "x3", "x4" ]
schema = StructType ( [ StructField(colname, FloatType(), False) for colname in colnames ] )

# assembler group all x1..x2 into a single col called X
assembler = VectorAssembler( inputCols = colnames, outputCol="X" )
test = spark.read.csv('dataanalytics-course/td01/data/iris_bin.test', schema = schema)
test = assembler.transform(test) #group all x1..x2 into a single col called X

test = test.select("X")

print("Schema: ")
test.printSchema()

print("Data")
print(test.show())

pred = lrModel.transform(test)

pred.show(30)
