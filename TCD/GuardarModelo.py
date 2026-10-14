from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

spark = SparkSession.builder \
    .appName("EntrenamientoModeloMovie") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

ratings = spark.read.csv("ratings.csv", header=True, inferSchema=True)
(training, testing) = ratings.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating")
modelo = als.fit(training)

modelo.write().overwrite().save("modelo_als")

print("Modelo entrenado y guardado con éxito en 'modelo_als'")
spark.stop()