#!/usr/bin/env python3
"""
Census Income Classification Project

This script predicts whether an individual's income exceeds $50K/year based on 
census data. The pipeline includes Exploratory Data Analysis (EDA), data 
preprocessing, and the training and evaluation of both a Decision Tree 
Classifier and a Deep Neural Network.
"""

# =============================================================================
# 1. Imports and Environment Setup
# =============================================================================
import os
import time
import warnings

# Suppress TensorFlow C++ MIN_LOG_LEVEL warnings and general warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# TensorFlow / Keras
import tensorflow.keras as keras

# Set visual theme
sns.set_theme()


# =============================================================================
# 2. Custom Callbacks
# =============================================================================
class ProgBarLoggerNEpochs(keras.callbacks.Callback):
    """Custom Keras Callback to log training progress every N epochs."""
    def __init__(self, num_epochs: int, every_n: int = 50):
        super().__init__()
        self.num_epochs = num_epochs
        self.every_n = every_n
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n == 0:
            s = f'Epoch [{epoch + 1}/ {self.num_epochs}]'
            logs_s = [f'{k.capitalize()}: {v:.4f}' for k, v in logs.items()]
            s_list = [s] + logs_s
            print(', '.join(s_list))


# =============================================================================
# 3. Data Loading and EDA
# =============================================================================
def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset from the specified filepath."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data from {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: Could not find the dataset at {filepath}")
        return pd.DataFrame()

def perform_eda(df: pd.DataFrame):
    """Performs Exploratory Data Analysis and prints key statistics."""
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*70)
    
    print(f"\nDataset Shape: {df.shape}")
    
    # Class Imbalance Check
    label_class_count = df['income_binary'].value_counts()
    print(f'\nValue count of the label column (income_binary):\n{label_class_count}')
    
    # Missing Values
    null_check = df.isnull().sum()
    print(f'\nColumns with null values:\n{null_check[null_check > 0]}')
    
    # Redundant Features / Correlation
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    # Note: plt.show() is commented out to prevent blocking the script execution. 
    # Uncomment if running interactively.
    # plt.show()


# =============================================================================
# 4. Data Preprocessing
# =============================================================================
def preprocess_data(df: pd.DataFrame):
    """Cleans, encodes, and splits the data for modeling."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)

    # 1. Label Encoding
    label_dict = {'<=50K': 0, '>50K': 1}
    df["income_binary"] = df["income_binary"].map(label_dict)

    # 2. Drop unnecessary features
    cols_to_drop = ['fnlwgt', 'sex_selfID', 'race', 'native-country', 'age', 'education-num']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # 3. Impute Missing Values
    str_cols = list(df.select_dtypes(include='object').columns)
    num_cols = list(df.select_dtypes(include=['float64', 'int64']).columns)

    for column in df.columns:
        if df[column].isnull().sum() > 0:
            if column in num_cols:
                df[column].fillna(value=df[column].mean(), inplace=True)
            elif column in str_cols:
                df[column].fillna(value=df[column].mode()[0], inplace=True)

    # 4. Feature Mapping
    occupation_map = {
        'Exec-managerial': 'Professional_Exec', 'Prof-specialty': 'Professional_Exec', 'Tech-support': 'Professional_Exec',
        'Sales': 'Sales_Admin', 'Adm-clerical': 'Sales_Admin',
        'Craft-repair': 'Service_Labor', 'Machine-op-inspct': 'Service_Labor', 'Transport-moving': 'Service_Labor',
        'Handlers-cleaners': 'Service_Labor', 'Farming-fishing': 'Service_Labor', 'Protective-serv': 'Service_Labor',
        'Priv-house-serv': 'Service_Labor', 'Other-service': 'Service_Labor',
        'Armed-Forces': 'Military'
    }
    if 'occupation' in df.columns:
        df['occupation'] = df['occupation'].map(occupation_map)

    education_map = {
        'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8,
        'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14,
        'Prof-school': 15, 'Doctorate': 16
    }
    if 'education' in df.columns:
        df['education'] = df['education'].map(education_map)

    # 5. One-Hot Encoding
    to_encode = ['workclass', 'marital-status', 'relationship', 'occupation']
    df_encode = pd.get_dummies(df[[col for col in to_encode if col in df.columns]])
    df = df.drop(columns=[col for col in to_encode if col in df.columns]).join(df_encode)

    # 6. Outlier Clipping (Winsorization)
    for col in ['capital-gain', 'capital-loss']:
        if col in df.columns:
            cap_val = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap_val)

    # 7. Train/Test Split
    X = df.drop(columns=['income_binary'])
    y = df['income_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=1234)
    
    print(f"Training Features Shape: {X_train.shape}")
    print(f"Testing Features Shape: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test


# =============================================================================
# 5. Model Training and Evaluation
# =============================================================================
def train_decision_tree(X_train, X_test, y_train, y_test):
    """Trains and evaluates a Decision Tree model, including GridSearch."""
    print("\n" + "="*70)
    print("DECISION TREE CLASSIFIER")
    print("="*70)

    # Grid Search CV
    print("Running Grid Search CV...")
    param_grid = {
        'max_depth': [2**n for n in range(2, 8)], 
        'min_samples_leaf': [25*2**n for n in range(0, 3)]
    }
    
    grid = GridSearchCV(DecisionTreeClassifier(class_weight='balanced'), param_grid, cv=5, scoring='f1')
    grid_search = grid.fit(X_train, y_train)
    
    print(f"Optimal Hyperparameters: {grid_search.best_params_}")
    
    # Train Final Model
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='binary')
    
    print(f"Accuracy Score: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    print("\nConfusion Matrix:")
    c_m = confusion_matrix(y_test, predictions, labels=[1, 0])
    print(pd.DataFrame(c_m, columns=['Predicted: >50K', 'Predicted: <=50K'], index=['Actual: >50K', 'Actual: <=50K']))
    
    return acc, f1

def train_neural_network(X_train, X_test, y_train, y_test):
    """Scales data, builds, trains, and evaluates a Deep Neural Network."""
    print("\n" + "="*70)
    print("DEEP NEURAL NETWORK")
    print("="*70)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Network Architecture
    n_features = X_train_scaled.shape[1]
    
    nn_model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(n_features,), name='input'),
        keras.layers.Dense(units=32, activation="relu", name="hl_1"),
        keras.layers.Dense(units=16, activation="relu", name="hl_2"),
        keras.layers.Dense(units=1, activation="sigmoid", name="output")
    ])
    
    nn_model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.1), 
        loss=keras.losses.BinaryCrossentropy(from_logits=False), 
        metrics=['accuracy']
    )
    
    print("\nTraining Model...")
    t0 = time.time()
    num_epochs = 18
    
    history = nn_model.fit(
        X_train_scaled, y_train, 
        epochs=num_epochs, 
        verbose=0, 
        validation_split=0.2, 
        callbacks=[ProgBarLoggerNEpochs(num_epochs, every_n=5)]
    )
    
    print(f"Training completed in {time.time()-t0:.2f} seconds.")
    
    # Predictions
    nn_probabilities = nn_model.predict(X_test_scaled, verbose=0)
    nn_predictions = (nn_probabilities >= 0.5).astype(int)
    
    # Evaluation
    acc = accuracy_score(y_test, nn_predictions)
    f1 = f1_score(y_test, nn_predictions, average='binary')
    
    print(f"Accuracy Score: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return acc, f1


# =============================================================================
# 6. Main Execution Block
# =============================================================================
if __name__ == "__main__":
    # Define File Path
    census_filepath = os.path.join(os.getcwd(), "data_capstone", "censusData.csv")
    
    # Execute Pipeline
    df = load_data(census_filepath)
    
    if not df.empty:
        perform_eda(df)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        dt_acc, dt_f1 = train_decision_tree(X_train, X_test, y_train, y_test)
        nn_acc, nn_f1 = train_neural_network(X_train, X_test, y_train, y_test)
        
        print("\n" + "="*70)
        print("FINAL RESULTS COMPARISON")
        print("="*70)
        results = pd.DataFrame({
            'Metric': ['Accuracy', 'F1 Score'],
            'DT Model': [dt_acc, dt_f1],
            'Neural Network': [nn_acc, nn_f1]
        })
        print(results.to_string(index=False))