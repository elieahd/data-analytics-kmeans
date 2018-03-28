from pyspark.ml.feature import StringIndexer
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StructField
from pyspark.sql.types import StructType
from pyspark.sql.types import FloatType
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.master("local").appName("kdd-cup").getOrCreate()

elementString = [2,3,4,42]
elementInteger = [1,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,32,33]
elementFloat = [25,26,27,28,29,30,31,34,35,36,37,38,39,40,41]

structFields = []
i = 0
while i < 42:
    i += 1
    if i in elementString:
        dataType = StringType()
    elif i in elementInteger:
        dataType = IntegerType()
    elif i in elementFloat:
        dataType = FloatType()
    colname = "x" + str(i)
    structFields.append(StructField(colname, dataType))

schema = StructType(structFields)
training = spark.read.csv('dataanalytics/td01/data/kddcup.data_10_percent', schema = schema)
training = training.withColumn("Y", (training.x42 == "normal.").cast(FloatType()))

for i in [2,3,4]:
    indexer = StringIndexer(inputCol=("x"+str(i)), outputCol=("x"+str(i)+"_index"))
    training = indexer.fit(training).transform(training)
