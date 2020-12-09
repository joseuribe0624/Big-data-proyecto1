#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark.sql.functions as F
spark = SparkSession.builder.config("spark.driver.memory", "4g").appName("twicth").getOrCreate()
sqlContext = SQLContext(spark)


# In[2]:


from pyspark.sql.functions import col, explode, array, lit
from pyspark.mllib.evaluation import MulticlassMetrics
import json
import matplotlib.pyplot as plt


# In[3]:


from pyspark.sql.types import *
schema = StructType([StructField('_id', StringType(),True),
                    StructField('_labels', StringType(), True),
                    StructField('betweenness', StringType(), True),
                    StructField('days', StringType(), True),
                    StructField('id', StringType(), True),
                    StructField('louvain', StringType(), True),
                    StructField('mature', StringType(), True),
                    StructField('node2vec', StringType(), True),
                    StructField('pageRank', StringType(), True),
                    StructField('partner', StringType(), True),
                    StructField('triangles', StringType(), True),
                    StructField('views', StringType(), True),
                    StructField('_start', StringType(), True),
                    StructField('_end', StringType(), True),
                    StructField('_type', StringType(), True)])
           

df = spark.read.format("csv").option("header",True).option("delimeter", ",")                        .schema(schema).load("dataset_topo.csv")


# # Cleaning

# In[4]:


columns_to_drop = ["_start", "_end", "_type", "_id", "_labels", "id"]
df = df.drop(*columns_to_drop)


# In[5]:


numeric_features = ["betweenness", "days", "louvain", "pageRank", "triangles", "views"]
categorical_features = ["mature", "partner"]
vectorial_features = ["node2vec"]


# In[6]:


for feature in numeric_features:
    df = df.withColumn(feature, col(feature).cast(DoubleType()))


# In[7]:


for feature in categorical_features:
    df = df.withColumn(feature, F.when(F.col(feature) == 'True', 1).when(F.col(feature) == 'False', 0).otherwise(F.col(feature).cast(IntegerType())))


# In[8]:


def parse_embedding_from_string(x):
    res = json.loads(x)
    res = [float(x_) for x_ in res]
    return res

retrieve_embedding = F.udf(parse_embedding_from_string, ArrayType(DoubleType()))


# In[9]:


for feature in vectorial_features:
    df = df.withColumn(feature, retrieve_embedding(F.col(feature)))


# In[10]:


print((df.count(), len(df.columns)))


# In[11]:


df.printSchema()


# ## 'count','mean','std','min','25%','50%','75%','max'

# In[12]:


df.summary().select(*numeric_features).show()


# In[13]:


df.summary().select(*categorical_features).show()


# ## Correlation Matrix

# In[14]:


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


# ## Transforming to vector

# In[15]:


from pyspark.sql.functions import expr


# In[16]:


arr_size = 2

df = df.select(["betweenness", "days", "louvain", "mature", "pageRank", "partner", "triangles", "views"]+[expr('node2vec[' + str(x) + ']') for x in range(arr_size)])


# In[17]:


df.printSchema()


# In[18]:


numeric_features = ["betweenness", "days", "louvain", "pageRank", "triangles", "views"]
categorical_features = ["mature", "partner"]
vectorial_features = ["node2vec"]


# In[19]:


features = numeric_features + categorical_features + ["node2vec[0]", "node2vec[1]"]
features.remove("mature")


# In[20]:


from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")
v_df = vectorAssembler.transform(df)
v_df = v_df.select(["features", "mature"])
v_df = v_df.withColumnRenamed("mature", "label")


# ## Oversampling

# In[21]:


major_df = v_df.filter(col("label") == 0)
minor_df = v_df.filter(col("label") == 1)
ratio = int(major_df.count()/minor_df.count()) + 1
print("ratio: {}".format(ratio))


# In[22]:


a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows combined_df = major_df.unionAll(oversampled_df)
combined_df = major_df.unionAll(oversampled_df)


# In[23]:


combined_df.groupBy("label").count().show()


# In[24]:


v_df = combined_df


# In[112]:


splits = v_df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]


# # Results with topologic features

# ## Logistic Regression

# In[26]:


from pyspark.ml.classification import LogisticRegression


# In[106]:


lr = LogisticRegression(maxIter=1000, regParam=0.03, elasticNetParam=0.3)


# In[107]:


lrModel = lr.fit(train_df)


# In[108]:


predictions = lrModel.transform(test_df)


# In[109]:


predictions_raw = predictions.withColumn("prediction", col("prediction"))                              .withColumn("label", col("label").cast(DoubleType()))                              .drop('features')                              .drop('rawPrediction')                              .drop('probability')


# In[110]:


metrics = MulticlassMetrics(predictions_raw.rdd)


# In[111]:


print("Summary Stats for LogisticRegression")
print("Accuracy = %s" % metrics.accuracy)
print("Precision = %s" % metrics.weightedPrecision)
print("Recall = %s" % metrics.weightedRecall)
print("F1 Score = %s" % metrics.weightedFMeasure())


# ## RandomForestClassifier

# In[57]:


from pyspark.ml.classification import RandomForestClassifier


# In[138]:


rf = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=20, numTrees=21)


# In[139]:


rf_model = rf.fit(train_df)


# In[140]:


rf_predictions = rf_model.transform(test_df)


# In[141]:


rf_predictions_raw = rf_predictions.withColumn("prediction", col("prediction"))                              .withColumn("label", col("label").cast(DoubleType()))                              .drop('features')                              .drop('rawPrediction')                              .drop('probability')


# In[142]:


rf_metrics = MulticlassMetrics(rf_predictions_raw.rdd)


# In[143]:


print("Summary Stats for RandomForestClassifier")
print("Accuracy = %s" % rf_metrics.accuracy)
print("Precision = %s" % rf_metrics.weightedPrecision)
print("Recall = %s" % rf_metrics.weightedRecall)
print("F1 Score = %s" % rf_metrics.weightedFMeasure())


# In[144]:


importances_features = rf_model.featureImportances.toArray().tolist()


# In[145]:


fig = plt.figure(figsize=(9, 6))
ax = fig.add_axes([0,0,1,1])
labels = ['betweenness','days','louvain','pageRank','triangles','views','partner','node2vec[0]','node2vec[1]']
ax.bar(labels, importances_features, width=0.7)
plt.show()


# ## Results without topologic features

# In[42]:


df_notopo = df.select(["days", "mature", "partner", "views"])


# In[43]:


vectorAssembler = VectorAssembler(inputCols=["days", "partner", "views"], outputCol="features")
v_df_notopo = vectorAssembler.transform(df_notopo)
v_df_notopo = v_df_notopo.select(["features", "mature"])
v_df_notopo = v_df_notopo.withColumnRenamed("mature", "label")


# In[44]:


major_df = v_df_notopo.filter(col("label") == 0)
minor_df = v_df_notopo.filter(col("label") == 1)
ratio = int(major_df.count()/minor_df.count()) + 1
print("ratio: {}".format(ratio))


# In[45]:


a = range(ratio)
# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows combined_df = major_df.unionAll(oversampled_df)
combined_df = major_df.unionAll(oversampled_df)


# In[46]:


combined_df.groupBy("label").count().show()


# In[47]:


v_df_notopo = combined_df


# In[48]:


splits = v_df_notopo.randomSplit([0.7, 0.3])
train_df_notopo = splits[0]
test_df_notop = splits[1]


# In[146]:


rf_notopo = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=20, numTrees=21)
rf_notopo_model = rf_notopo.fit(train_df_notopo)
rf_notopo_predictions = rf_notopo_model.transform(test_df_notop)
rf_notopo_predictions_raw = rf_notopo_predictions.withColumn("prediction", col("prediction"))                              .withColumn("label", col("label").cast(DoubleType()))                              .drop('features')                              .drop('rawPrediction')                              .drop('probability')
rf_notopo_metrics = MulticlassMetrics(rf_notopo_predictions_raw.rdd)
print("Summary Stats for RandomForestClassifier")
print("Accuracy = %s" % rf_notopo_metrics.accuracy)
print("Precision = %s" % rf_notopo_metrics.weightedPrecision)
print("Recall = %s" % rf_notopo_metrics.weightedRecall)
print("F1 Score = %s" % rf_notopo_metrics.weightedFMeasure())


# In[147]:


importances_features = rf_notopo_model.featureImportances.toArray().tolist()
fig = plt.figure(figsize=(9, 6))
ax = fig.add_axes([0,0,1,1])
labels = ["days", "partner", "views"]
ax.bar(labels, importances_features, width=0.7)
plt.show()

