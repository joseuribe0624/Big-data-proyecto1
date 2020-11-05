#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql.types import StructType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
import pyspark.sql.functions as f
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import DoubleType


# In[ ]:


from pyspark.sql.types import *
spark = SparkSession.builder.appName("Mlseverity").getOrCreate()

accidentsSchema = StructType([
                    StructField("TMC", DoubleType(), True),
                    StructField("Severity", DoubleType(), True),
                    StructField("Start_Lat", DoubleType(), True),
                    StructField("Start_Lng", DoubleType(), True),
                    StructField("Distance(mi)", DoubleType(), True),
                    StructField("Temperature(F)", DoubleType(), True),
                    StructField("Wind_Chill(F)", DoubleType(), True),
                    StructField("Humidity(%)", DoubleType(), True),
                    StructField("Pressure(in)", DoubleType(), True),
                    StructField("Visibility(mi)", DoubleType(), True),
                    StructField("Wind_Speed(mph)", DoubleType(), True),
                    StructField("Precipitation(in)", DoubleType(), True),
                    StructField("Duration", DoubleType(), True),
                    StructField("Side", DoubleType(), True),
                    StructField("City", DoubleType(), True),
                    StructField("County", DoubleType(), True),
                    StructField("State", DoubleType(), True),
                    StructField("Wind_Direction", DoubleType(), True),
                    StructField("Weather_Condition", DoubleType(), True),
                    StructField("Amenity", DoubleType(), True),
                    StructField("Bump", DoubleType(), True),
                    StructField("Crossing", DoubleType(), True),
                    StructField("Give_Way", DoubleType(), True),
                    StructField("Junction", DoubleType(), True),
                    StructField("No_Exit", DoubleType(), True),
                    StructField("Railway", DoubleType(), True),
                    StructField("Roundabout", DoubleType(), True),
                    StructField("Station", DoubleType(), True),
                    StructField("Stop", DoubleType(), True),
                    StructField("Traffic_Calming", DoubleType(), True),
                    StructField("Traffic_Signal", DoubleType(), True),
                    StructField("Turning_Loop", DoubleType(), True),
                    StructField("Civil_Twilight", DoubleType(), True),
                ])

dfCSV = spark.readStream.option("delimeter", ",").option("header", True).schema(accidentsSchema).csv("tmp")


# In[ ]:


features = ["TMC", "Start_Lat", "Start_Lng", "Distance(mi)", "Temperature(F)", "Wind_Chill(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)", "Duration", "Side", "City", "County", "State", "Wind_Direction", "Weather_Condition", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", "Civil_Twilight"]

dfCSV = dfCSV.withColumnRenamed("Severity", "label")
vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")


# In[ ]:


lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)


# In[ ]:


pipeline = Pipeline(stages=[vectorAssembler, lr])


# In[ ]:


splits = dfCSV.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# In[ ]:


def foreach_batch_function(df, epoch_id):
    train_df = splits[0]
    test_df = splits[1]
    
    fited_pipeline = pipeline.fit(train_df)
    predictions = fited_pipeline.transform(test_df)
    
    predictions_raw = predictions.withColumn("prediction", col("prediction"))                              .withColumn("label", col("label").cast(DoubleType()))                              .drop('features')                              .drop('rawPrediction')                              .drop('probability')
    
    metrics = MulticlassMetrics(predictions_raw.rdd)
    
    print("Summary Stats for LogisticRegression")
    print("Accuracy = %s" % metrics.accuracy)
    print("Precision = %s" % metrics.weightedPrecision)
    print("Recall = %s" % metrics.weightedRecall)
    print("F1 Score = %s" % metrics.weightedFMeasure()) 


# In[ ]:


stream = dfCSV.writeStream.foreachBatch(foreach_batch_function).start()
stream.awaitTermination()


# In[ ]:


##persistedModel = LogisticRegressionModel.load("LogisticRegression")


# In[ ]:


##predictions = persistedModel.transform(v_df)


# In[ ]:


"""predictions.createOrReplaceTempView("df")

prueba = spark.sql("SELECT prediction as Severiry, count(*) as N  FROM df GROUP BY prediction")

query = prueba.writeStream.outputMode("complete").format("console").start()
query.awaitTermination()"""


# In[ ]:




