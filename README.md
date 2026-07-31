# Census Income Classification & Model Comparison

An end-to-end machine learning pipeline designed to predict individual income levels (above or below $50K) using census data, comparing the performance and interpretability of **Decision Tree** and **Neural Network** models.

## 📊 Project Overview
Predicting individual income from demographic and employment census data is a classic machine learning classification challenge. This project explores the full data science lifecycle—from exploratory data analysis and rigorous preprocessing to model training, evaluation, and comparative analysis between traditional rule-based algorithms (Decision Trees) and deep learning architectures (Neural Networks).

## 🚀 Key Features
* **Data Preprocessing & Cleaning:** Handled missing values, encoded categorical variables, and standardized numerical features to prepare clean data inputs.
* **Model Implementation:** Trained and tuned both Decision Tree classifiers and Multi-Layer Perceptron (Neural Network) models.
* **Comparative Evaluation:** Evaluated models using key metrics including Accuracy, Precision, Recall, and ROC-AUC to compare performance trade-offs between interpretability and predictive power.

## 💻 Tech Stack
* **Language:** Python
* **Libraries & Frameworks:** Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, tensorflow.keras, scipy.stats, time
## 📊 Dataset
This project utilizes the **Adult (Census Income)** dataset donated to the UCI Machine Learning Repository. 

While raw data files have been omitted from this repository to maintain clean version control, you can access, explore, or download the dataset directly from the [UCI Machine Learning Repository - Adult Dataset Page](https://archive.ics.uci.edu/dataset/2/adult).

### Programmatic Data Fetching
Rather than storing static data files, the accompanying Python scripts dynamically fetch the dataset using the official repository package:

```python
from ucimlrepo import fetch_ucirepo 

# Fetch dataset programmatically
adult = fetch_ucirepo(id=2) 
X = adult.data.features 
y = adult.data.targets
