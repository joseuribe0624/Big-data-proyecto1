import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import *


spark = SparkSession.builder.appName('us_accidents').getOrCreate()
sqlContext =SQLContext(spark)

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
           
           
df = spark.read.format("csv").option("header",True).option("delimeter", ",")\
                        .schema(schema).load("C:/Users/joseuribe/Desktop/universidad/bigdata/proyecto/us_accidents.csv")



def cant_Col_Rows():
	#data count columns and registers
	print((df.count(), len(df.columns)))

def printSummary():
	df.summary().select('ID','Source','TMC','Severity','Start_Time').show()
	df.summary().select('End_Time','Start_Lat','End_Lat','End_Lng','Distance(mi)').show()
	df.summary().select('Description','Number','Street','Side','City').show()
	df.summary().select('County','State','Zipcode','Country','TimeZone').show()
	df.summary().select('Airport_Code','Weather_Timestamp','Temperature(F)','Wind_Chill(F)','Humidity(%)').show()
	df.summary().select('Pressure(in)','Visibility(mi)', 'Wind_Direction','Wind_Speed(mph)','Precipitation(in)').show()
	df.summary().select('Weather_Condition','Amenity','Bump','Crossing','Give_Way').show()
	df.summary().select('Junction','No_Exit','Railway','Roundabout','Station').show()
	df.summary().select('Stop','Traffic_Calming','Traffic_Signal','Turning_Loop', 'Sunrise_Sunset').show()
	df.summary().select('Civil_Twilight','Nautical_Twilight','Astronomical_Twilight').show()


def setNullsToMean():
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


def CountsValues():
     df2 = df.select([F.count(F.when(F.isnan(c) | F.isnull(c), c)).alias(c) for (c,c_type) in df.dtypes]).toPandas()

     if len(df2) == 0:
         print("There are no any missing values!")
         
     print(df2.rename(index={0: 'count'}).T.sort_values("count",ascending=False))



def compute_correlation_matrix(df, method='pearson'):
     df_rdd = df.rdd.map(lambda row: row[0:])
     corr_mat = Statistics.corr(df_rdd, method=method)
     corr_mat_df = pd.DataFrame(corr_mat,
                    columns=df.columns, 
                    index=df.columns)
     return corr_mat_df

def CorrMatrix():
     numeric_features = [t[0] for t in df.dtypes if t[1] == 'int' or t[1]=='double']
     df2=df.select(numeric_features)
     print(compute_correlation_matrix(df2))



