from pyspark.sql import SparkSession
from pyspark.sql.functions import lower, col, concat_ws, udf
from pyspark.sql.types import StringType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark
spark = SparkSession.builder.appName("FakeNewsClassification").getOrCreate()

# ==========================
# Task 1: Load & Exploration
# ==========================
df = spark.read.csv("fake_news_sample.csv", header=True, inferSchema=True)
df.createOrReplaceTempView("news_data")

df.show(5)
print(f"Total articles: {df.count()}")
df.select("label").distinct().show()

df.limit(5).write.csv("task1_output.csv", header=True, mode="overwrite")

# ==========================
# Task 2: Preprocessing
# ==========================
df = df.withColumn("text", lower(col("text")))
tokenizer = Tokenizer(inputCol="text", outputCol="words")
tokenized_df = tokenizer.transform(df)

remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
cleaned_df = remover.transform(tokenized_df)

task2_df = cleaned_df.select("id", "title", "filtered_words", "label")

# Convert array column to string for CSV writing
task2_df.withColumn("filtered_words_str", concat_ws(" ", "filtered_words")) \
    .select("id", "title", "filtered_words_str", "label") \
    .write.csv("task2_output.csv", header=True, mode="overwrite")

# ==========================
# Task 3: Feature Extraction
# ==========================
hashingTF = HashingTF(inputCol="filtered_words", outputCol="rawFeatures", numFeatures=10000)
tf_df = hashingTF.transform(task2_df)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(tf_df)
tfidf_df = idf_model.transform(tf_df)

indexer = StringIndexer(inputCol="label", outputCol="label_index")
final_df = indexer.fit(tfidf_df).transform(tfidf_df)

# Convert vector column to string for CSV writing
vector_to_string = udf(lambda v: str(v), StringType())

task3_df = final_df.withColumn("filtered_words_str", concat_ws(" ", "filtered_words")) \
                   .withColumn("features_str", vector_to_string(col("features"))) \
                   .select("id", "filtered_words_str", "features_str", "label_index")

task3_df.write.csv("task3_output.csv", header=True, mode="overwrite")

# ==========================
# Task 4: Train & Predict
# ==========================
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features", labelCol="label_index")
model = lr.fit(train_df)

predictions = model.transform(test_df)

# Join with titles for display
titles_df = df.select("id", "title")
final_preds = predictions.select("id", "label_index", "prediction").join(titles_df, "id")

final_preds.select("id", "title", "label_index", "prediction") \
    .write.csv("task4_output.csv", header=True, mode="overwrite")

# ==========================
# Task 5: Evaluation
# ==========================
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="accuracy")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction", metricName="f1")

accuracy = evaluator_acc.evaluate(predictions)
f1_score = evaluator_f1.evaluate(predictions)

metrics_df = spark.createDataFrame([
    ("Accuracy", accuracy),
    ("F1 Score", f1_score)
], ["Metric", "Value"])

metrics_df.show()
metrics_df.write.csv("task5_output.csv", header=True, mode="overwrite")
