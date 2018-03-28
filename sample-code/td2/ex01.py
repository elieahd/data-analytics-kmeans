from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StringType
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import col

spark = SparkSession.builder.master("local").appName("iris").getOrCreate()

colnames = ["x1", "x2", "x3", "x4", "label" ]
schemaArray = [StructField(colname, DoubleType(), False) for colname in colnames[:-1]]
schemaArray.append(StructField(colnames[4], StringType(), False))
schema = StructType (schemaArray)
data = spark.read.csv('dataanalytics/td01/data/iris_clustering.dat', schema=schema)

assembler = VectorAssembler(inputCols = ["x1","x2","x3","x4"], outputCol="features")
data = assembler.transform(data);

data = data.select(monotonically_increasing_id().alias('p_id'), 'features', 'label')

kmeans = KMeans().setK(3)
model = kmeans.fit(data)
data = model.transform(data)
data.show(100)

dataPrediction = data.groupBy('prediction').count()
dataPrediction = dataPrediction.withColumnRenamed('count','total-predicition')
dataCounted = data.groupBy('prediction', 'label').count().withColumnRenamed('count','total-label')
avg = dataPrediction.join(dataCounted, 'prediction').withColumn('mean', (col('total-label') * 100 /col('total-predicition')))





