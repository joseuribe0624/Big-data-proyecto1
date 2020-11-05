#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark


# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark.sql.functions as F
spark = SparkSession.builder.appName('us_accidents').getOrCreate()
sqlContext =SQLContext(spark)


# In[3]:


from pyspark.sql.types import *
schema = StructType([StructField('ID', StringType(),True),
                    StructField('Source', StringType(), True),
                    StructField('TMC', DoubleType(), True),
                    StructField('Severity', IntegerType(), True),
                    StructField('Start_Time', StringType(), True),
                    StructField('End_Time', StringType(), True),
                    StructField('Start_Lat', DoubleType(), True),
                    StructField('Start_Lng', DoubleType(), True),
                    StructField('End_Lat', DoubleType(), True),
                    StructField('End_Lng', DoubleType(), True),
                    StructField('Distance(mi)', DoubleType(), True),
                    StructField('Description', StringType(), True),
                    StructField('Number', DoubleType(), True),
                    StructField('Street', StringType(), True),
                    StructField('Side', StringType(), True),
                    StructField('City', StringType(), True),
                    StructField('County', StringType(), True),
                    StructField('State', StringType(), True),
                    StructField('Zipcode', StringType(), True),
                    StructField('Country', StringType(), True),
                    StructField('TimeZone', StringType(), True),
                    StructField('Airport_Code', StringType(), True),
                    StructField('Weather_Timestamp', StringType(), True),
                    StructField('Temperature(F)', DoubleType(), True),
                    StructField('Wind_Chill(F)', DoubleType(), True),
                    StructField('Humidity(%)', DoubleType(), True),
                    StructField('Pressure(in)', DoubleType(), True),
                    StructField('Visibility(mi)', DoubleType(), True),
                    StructField('Wind_Direction', StringType(), True),
                    StructField('Wind_Speed(mph)', DoubleType(), True),
                    StructField('Precipitation(in)', DoubleType(), True),
                    StructField('Weather_Condition', StringType(), True),
                    StructField('Amenity', StringType(), True),
                    StructField('Bump', StringType(), True),
                    StructField('Crossing', StringType(), True),
                    StructField('Give_Way', StringType(), True),
                    StructField('Junction', StringType(), True),
                    StructField('No_Exit', StringType(), True),
                    StructField('Railway', StringType(), True),
                    StructField('Roundabout', StringType(), True),
                    StructField('Station', StringType(), True),
                    StructField('Stop', StringType(), True),
                    StructField('Traffic_Calming', StringType(), True),
                    StructField('Traffic_Signal', StringType(), True),
                    StructField('Turning_Loop', StringType(), True),
                    StructField('Sunrise_Sunset', StringType(), True),
                    StructField('Civil_Twilight', StringType(), True),
                    StructField('Nautical_Twilight', StringType(), True),
                    StructField('Astronomical_Twilight', StringType(), True)])
           

df = spark.read.format("csv").option("header",True).option("delimeter", ",")                        .schema(schema).load("US_Accidents_June20.csv")


# In[4]:


print((df.count(), len(df.columns)))


# ## 'count','mean','std','min','25%','50%','75%','max'

# In[5]:


df.summary().select('ID','Source','TMC','Severity','Start_Time').show()


# In[6]:


df.summary().select('End_Time','Start_Lat','End_Lat','End_Lng','Distance(mi)').show()


# In[7]:


df.summary().select('Description','Number','Street','Side','City').show()


# In[8]:


df.summary().select('County','State','Zipcode','Country','TimeZone').show()


# In[9]:


df.summary().select('Airport_Code','Weather_Timestamp','Temperature(F)','Wind_Chill(F)','Humidity(%)').show()


# In[10]:


df.summary().select('Pressure(in)','Visibility(mi)', 'Wind_Direction','Wind_Speed(mph)','Precipitation(in)').show()


# In[11]:


"""
Counts number of nulls and nans in each column
"""
df2 = df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in df.dtypes]).toPandas()

if len(df2) == 0:
    print("There are no any missing values!")
    
print(df2.rename(index={0: 'count'}).T.sort_values("count",ascending=False))


# ## Convert null to mean (numeric features)

# In[4]:


df = df.withColumn("Number", F.when(F.col("Number").isNull(), 5975.3826).otherwise(F.col("Number")))
df = df.withColumn("TMC", F.when(F.col("TMC").isNull(), 208.0225).otherwise(F.col("TMC")))
df = df.withColumn("End_Lat", F.when(F.col("End_Lat").isNull(), 37.5575).otherwise(F.col("End_Lat")))
df = df.withColumn("End_Lng", F.when(F.col("End_Lng").isNull(), -100.455).otherwise(F.col("End_Lng")))
df = df.withColumn("Temperature(f)", F.when(F.col("Temperature(f)").isNull(), 61.93511).otherwise(F.col("Temperature(f)")))
df = df.withColumn("Wind_chill(f)", F.when(F.col("Wind_chill(f)").isNull(), 53.5572).otherwise(F.col("Wind_chill(f)")))
df = df.withColumn("Humidity(%)", F.when(F.col("Humidity(%)").isNull(), 65.1142).otherwise(F.col("Humidity(%)")))
df = df.withColumn("Pressure(in)", F.when(F.col("Pressure(in)").isNull(), 29.7446).otherwise(F.col("Pressure(in)")))
df = df.withColumn("Visibility(mi)", F.when(F.col("Visibility(mi)").isNull(), 9.1226).otherwise(F.col("Visibility(mi)")))
df = df.withColumn("Wind_Speed(mph)", F.when(F.col("Wind_Speed(mph)").isNull(), 8.21902).otherwise(F.col("Wind_Speed(mph)")))
df = df.withColumn("Precipitation(in)", F.when(F.col("Precipitation(in)").isNull(), 0.01598).otherwise(F.col("Precipitation(in)")))


# ## Correlation Matrix

# In[9]:


from pyspark.mllib.stat import Statistics
import pandas as pd


# result can be used w/ seaborn's heatmap
def compute_correlation_matrix(df, method='pearson'):
    # wrapper around
    # https://forums.databricks.com/questions/3092/how-to-calculate-correlation-matrix-with-all-colum.html
    df_rdd = df.rdd.map(lambda row: row[0:])
    corr_mat = Statistics.corr(df_rdd, method=method)
    corr_mat_df = pd.DataFrame(corr_mat,
                    columns=df.columns, 
                    index=df.columns)
    return corr_mat_df

numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1]=='double']
df2=df.select(numeric_features)
print(df2)

print(compute_correlation_matrix(df2))


# ## Convert null to mode (No-numeric features)

# In[5]:


Description = df.groupby("Description").count().orderBy("count", ascending=False).first()[0]
City = df.groupby("City").count().orderBy("count", ascending=False).first()[0]
Zipcode = df.groupby("Zipcode").count().orderBy("count", ascending=False).first()[0]    
TimeZone = df.groupby("TimeZone").count().orderBy("count", ascending=False).first()[0]
Airport_Code = df.groupby("Airport_Code").count().orderBy("count", ascending=False).first()[0]
Weather_Timestamp = df.groupby("Weather_Timestamp").count().orderBy("count", ascending=False).first()[0]
Wind_Direction = df.groupby("Wind_Direction").count().orderBy("count", ascending=False).first()[0] 
Weather_Condition = df.groupby("Weather_Condition").count().orderBy("count", ascending=False).first()[0]   
Station = df.groupby("Station").count().orderBy("count", ascending=False).first()[0]
Sunrise_Sunset = df.groupby("Sunrise_Sunset").count().orderBy("count", ascending=False).first()[0] 
Civil_Twilight = df.groupby("Civil_Twilight").count().orderBy("count", ascending=False).first()[0]
Nautical_Twilight = df.groupby("Nautical_Twilight").count().orderBy("count", ascending=False).first()[0]
Astronomical_Twilight = df.groupby("Astronomical_Twilight").count().orderBy("count", ascending=False).first()[0]

df = df.withColumn("Description", F.when(F.col("Description").isNull(), Description).otherwise(F.col("Description")))
df = df.withColumn("City", F.when(F.col("City").isNull(), City).otherwise(F.col("City")))
df = df.withColumn("Zipcode", F.when(F.col("Zipcode").isNull(), Zipcode).otherwise(F.col("Zipcode")))
df = df.withColumn("TimeZone", F.when(F.col("TimeZone").isNull(), TimeZone).otherwise(F.col("TimeZone")))
df = df.withColumn("Airport_Code", F.when(F.col("Airport_Code").isNull(), Airport_Code).otherwise(F.col("Airport_Code")))
df = df.withColumn("Weather_Timestamp", F.when(F.isnan("Weather_Timestamp") | F.isnull("Weather_Timestamp"), Weather_Timestamp).otherwise(F.col("Weather_Timestamp")))
df = df.withColumn("Wind_Direction", F.when(F.col("Wind_Direction").isNull(), Wind_Direction).otherwise(F.col("Wind_Direction")))
df = df.withColumn("Weather_Condition", F.when(F.col("Weather_Condition").isNull(), Weather_Condition).otherwise(F.col("Weather_Condition")))
df = df.withColumn("Station", F.when(F.col("Station").isNull(), Station).otherwise(F.col("Station")))
df = df.withColumn("Sunrise_Sunset", F.when(F.col("Sunrise_Sunset").isNull(), Sunrise_Sunset).otherwise(F.col("Sunrise_Sunset")))
df = df.withColumn("Civil_Twilight", F.when(F.col("Civil_Twilight").isNull(), Civil_Twilight).otherwise(F.col("Civil_Twilight")))
df = df.withColumn("Nautical_Twilight", F.when(F.col("Nautical_Twilight").isNull(), Nautical_Twilight).otherwise(F.col("Nautical_Twilight")))
df = df.withColumn("Astronomical_Twilight", F.when(F.col("Astronomical_Twilight").isNull(), Astronomical_Twilight).otherwise(F.col("Astronomical_Twilight")))


# ## Transform no-numeric features

# In[6]:


no_numeric = ["Source", "Start_Time", "End_Time", "Description", "Street", "Side", "City", "County", "State", "Zipcode", "Country", "TimeZone", "Airport_Code", "Weather_Timestamp", "Wind_Direction", "Weather_Condition", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
numeric_features = ["TMC", "Severity", "Start_Lat", "Start_Lng", "End_Lat", "End_Lng", "Distance(mi)", "Number", "Temperature(f)", "Wind_chill(f)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"]
output = "Severity"


# ### Text embedding for Description column

# In[7]:


from pyspark.ml.feature import Word2Vec

df = df.withColumn("Description", F.split(df["Description"], " "))

word2Vec = Word2Vec(inputCol="Description", outputCol="DescriptionVec")
model = word2Vec.fit(df)

df = model.transform(df)


# ### OneHotEncoder

# In[8]:


fetures_toencode = ["Source", "Street", "Side", "City","County", "State", "Zipcode", "Country", "TimeZone", "Airport_Code", "Wind_Direction", "Weather_Condition", "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop", "Sunrise_Sunset", "Civil_Twilight", "Nautical_Twilight", "Astronomical_Twilight"]
fetures_encoded = [feature+"Vec" for feature in fetures_toencode]


# In[9]:


from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer

for index_feature in range(len(fetures_toencode)):
    indexer = StringIndexer(inputCol=fetures_toencode[index_feature], outputCol=fetures_encoded[index_feature])
    model = indexer.fit(df)
    df = model.transform(df)


# ### Dates

# In[10]:


features_dates = ["Start_Time", "End_Time"]


# In[11]:


from pyspark.sql.functions import col, unix_timestamp, to_date

for feature in features_dates:
    df = df.withColumn(feature+"Vec", unix_timestamp(feature, "yyyy-MM-dd HH:mm:ss"))


# In[12]:


df.printSchema()


# In[13]:


features = ["Airport_CodeVec", "AmenityVec", "Astronomical_TwilightVec", "BumpVec", "CityVec", "Civil_TwilightVec", "CountryVec", "CountyVec", "CrossingVec", "DescriptionVec", "Distance(mi)", "End_Lat", "End_Lng", "End_TimeVec", "Give_WayVec", "Humidity(%)", "JunctionVec", "Nautical_TwilightVec", "No_ExitVec", "Number", "Precipitation(in)", "Pressure(in)", "RailwayVec", "RoundaboutVec", "Severity", "SideVec", "SourceVec", "Start_Lat", "Start_Lng", "Start_TimeVec", "StateVec", "StationVec", "StopVec", "StreetVec", "Sunrise_SunsetVec", "Temperature(f)", "TimeZoneVec", "TMC", "Traffic_CalmingVec", "Traffic_SignalVec", "Turning_LoopVec", "Visibility(mi)", "Weather_ConditionVec", "Wind_chill(f)", "Wind_DirectionVec", "Wind_Speed(mph)", "ZipcodeVec"]


# In[14]:


from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")
v_df = vectorAssembler.transform(df)
v_df = v_df.select(["features", "Severity"])
v_df = v_df.withColumnRenamed("Severity","label")


# In[15]:


splits = v_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# ## Logistic Regression

# In[17]:


from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(train_df)


# In[42]:


predictions = lrModel.transform(test_df)


# In[46]:


predictions.select("prediction", "label", "features").show(5)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = %g " % (accuracy))


# In[48]:


print("Accuracy = %g " % (accuracy))


# ## Multilayer perceptron classifier

# In[53]:


from pyspark.ml.classification import MultilayerPerceptronClassifier



trainer = MultilayerPerceptronClassifier(maxIter=100, layers=[100, 100], blockSize=128, seed=1234)

model = trainer.fit(train_df)


# In[ ]:


predictions_mpc = model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions_mpc)
print("Accuracy = %g " % (accuracy))

