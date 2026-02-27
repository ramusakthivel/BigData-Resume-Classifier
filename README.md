
## Automated Resume Classification System

### Project Overview

An end-to-end Big Data and Deep Learning pipeline designed to categorize resumes into 20+ job sectors (e.g., HR, IT, Sales). This project demonstrates the integration of Distributed Storage (**Hadoop/HDFS**), Data Warehousing (**Hive**), and Neural Networks (**TensorFlow/Keras**).

### Technical Stack

* **Infrastructure:** Hadoop 3.x (HDFS & YARN), Apache Hive 2.3.9.
* **Data Processing:** Python 3.12, Pandas (NLP Preprocessing).
* **Machine Learning:** TensorFlow (Multi-Layer Perceptron), Scikit-Learn (TF-IDF Vectorization, Label Encoding).
* **Environment:** Ubuntu (WSL2), Linux Bash.

### Pipeline Stages

#### 1. Big Data Ingestion & Management

Raw resume data (~54,000 records) was loaded into **HDFS** and managed via **Apache Hive**. To handle complex CSV formatting (embedded HTML and commas), the **OpenCSVSerde** was utilized for accurate schema-on-read.

#### 2. NLP & Preprocessing

* **Cleaning:** Utilized Regex to remove HTML tags, URLs, and special characters.
* **Vectorization:** Implemented **TF-IDF** (Term Frequency-Inverse Document Frequency) to transform text into numerical feature vectors.

#### 3. Neural Network Architecture

The core "brain" is a **Multi-Layer Perceptron (MLP)**:

* **Input Layer:** 3000 features (TF-IDF tokens).
* **Hidden Layers:** Dense layers with **ReLU** activation and **Dropout** (0.3) for regularization.
* **Output Layer:** Softmax layer for multi-class classification.


### Performance & Key Insights

* **Accuracy:** Achieved **99.8% training accuracy**.
* **Validation:** Validation accuracy reached **63.5%**, indicating areas for further hyperparameter tuning to reduce overfitting.
* **Resource Optimization:** Successfully managed deployment within a resource-constrained Ubuntu environment (99% disk usage) by clearing `pip cache` and utilizing `tensorflow-cpu`.

### How to Run

1. **Hadoop Start:** `start-all.sh`
2. **Preprocess:** `python3 clean_resumes.py`
3. **Train Model:** `python3 train_mlp.py`
4. **Inference:** `python3 predict_resume.py`



### Troubleshooting Log

During development, several environment-specific challenges were overcome:

* **Java Gateway Errors:** Fixed PySpark-to-JVM connection issues by correctly mapping `JAVA_HOME` to OpenJDK 11.
* **Storage Constraints:** Resolved `OSError: [Errno 28]` by performing emergency space recovery on the WSL root filesystem.
