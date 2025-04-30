# Assignment-5-FakeNews-Detection 📰🔍

This project implements a complete Spark-based pipeline to detect fake news articles. It processes a labeled dataset using Spark SQL and MLlib, and evaluates classification performance using logistic regression.

---

## 📁 Dataset

**Input:** `fake_news_sample.csv`  
**Columns:** `id`, `title`, `text`, `label` (`FAKE` or `REAL`)

---

## ✅ Task Breakdown

| Task | Description |
|------|-------------|
| Task 1 | Load dataset, show schema, run basic SQL queries |
| Task 2 | Tokenize and clean text using Spark NLP tools |
| Task 3 | Extract TF-IDF features, index labels, assemble feature vectors |
| Task 4 | Train logistic regression model, generate predictions |
| Task 5 | Evaluate model using Accuracy and F1 Score |


### 🧪 Task 1: Load & Basic Exploration
- Load the dataset using Spark with schema inference.
- Create a temporary view `news_data`.
- Run:
  - Show 5 sample rows
  - Count articles
  - Distinct labels
- **Output:** `task1_output.csv`

---

### ✂️ Task 2: Text Preprocessing
- Convert all text to lowercase.
- Tokenize text using `Tokenizer`.
- Remove stopwords using `StopWordsRemover`.
- Optional: Register `cleaned_news` view.
- **Output:** `task2_output.csv`  
  Format: `id`, `title`, `filtered_words`, `label`

---

### ⚙️ Task 3: Feature Extraction
- Apply `HashingTF` + `IDF` on `filtered_words`.
- Convert `label` to `label_index` using `StringIndexer`.
- Assemble into single `features` column.
- **Output:** `task3_output.csv`  
  Format: `id`, `filtered_words`, `features`, `label_index`

---

### 🤖 Task 4: Model Training
- Split into train/test (80/20).
- Train `LogisticRegression` on `features`.
- Predict on test set.
- **Output:** `task4_output.csv`  
  Format: `id`, `title`, `label_index`, `prediction`

---

### 📊 Task 5: Model Evaluation
- Use `MulticlassClassificationEvaluator`
- Metrics:
  - Accuracy
  - F1 Score
- **Output:** `task5_output.csv`  
  Format: Metric,Value Accuracy,1.0 F1 Score,1.0

## ▶️ How to Execute

### 🔹 Step-by-Step Commands

✅ Make sure spark-submit is installed and fake_news_sample.csv is present in your directory.

```bash
# Step 1: (Optional) Generate Dataset
python Dataset_Generator.py

# Step 2: Run the Spark Pipeline
spark-submit FakeNewsPipeline.py
```


## 📌 Project Summary

This project builds a complete Spark MLlib pipeline to classify news articles as FAKE or REAL. It performs text preprocessing, TF-IDF feature extraction, logistic regression training, and model evaluation using Apache Spark.

Dataset: fake_news_sample.csv (500 rows, 250 FAKE, 250 REAL)

Output files: task1_output.csv to task5_output.csv (one per stage)

Key Tools: PySpark, MLlib, Tokenizer, StopWordsRemover, HashingTF, IDF, LogisticRegression
