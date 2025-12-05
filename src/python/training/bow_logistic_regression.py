################################################################################
# NLP CLASSIFICATION PIPELINE (Bag of Words + Logistic Regression)
# 1. Data Loading & Selection
# 2. Train/Test Split
# 3. Pipeline Construction (CountVectorizer + LogReg)
# 4. Model Training & Evaluation
################################################################################

# --- Imports ---
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pandas as pd
import time
# Scikit-learn modules
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)
# Custom utilities
from utils.data_tools import check_overfitting
# from utils.data_tools import underSample2Min # Uncomment if balancing is needed

# --- 1. Data Loading ---

try:
    # Load the dataset prepared in R
    df = pd.read_csv("data/processed/news_prepared.csv") # returns pandas.DataFrame
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: 'news_prepared.csv' not found. Please check the directory.")
    exit()

# --- 2. Feature Selection ---

# We use 'text_tfidf', which was cleaned in the R script (lowercase, no punctuation)
X = df['text_tfidf']  # Input features (pandas.Series)
y = df['is_fake']     # Target labels (pandas.Series)

# Ensure target is integer type
y = y.astype(int)

# Check class balance
# Class 0 (True) is approx 55%, so undersampling is not strictly necessary.
print("\n--- Class Balance (%) ---")
print(y.value_counts(normalize=True) * 100)

# --- 3. Train / Test Split ---

print(f'\nOriginal X shape: {X.shape}')
print(f'Original y shape: {y.shape}')

# Split the dataset into training and testing sets
# test_size=0.20: 80% for training, 20% for testing
# stratify=y: Maintains the same proportion of Fake/True news in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=0, 
    stratify=y 
)

# Inspect shapes of resulting subsets
print("\n--- Data Shapes after Split ---")
dataset_vars = {
    'X': X, 
    'y': y, 
    'X_train': X_train, 
    'X_test': X_test, 
    'y_train': y_train, 
    'y_test': y_test
}

for name, var in dataset_vars.items():
    print(f"{name}: {var.shape}")

# --- 4. Pipeline Construction ---

# Bag-of-Words approach: Convert text documents to fixed-length vectors of counts.
# This creates a Document-Term Matrix (DTM).
cp = Pipeline([
    ('vectorizer', TfidfVectorizer(norm='l2')),       # Converts text to token counts
    ('classifier', LogisticRegression())     # Probabilistic Linear Classifier
    # Alternative: ('classifier', LinearSVC())
])

# --- 5. Hyperparameter Configuration ---

# Configuration parameters for the pipeline components.
# Syntax: 'component_name__parameter_name'
clsfParams = {
    'classifier__C': 1.0,  
    'vectorizer__stop_words': 'english',  # Remove common words (the, is, at...)
    'vectorizer__ngram_range': (1, 1),    # (1,1) = Unigrams only. (1,2) = Unigrams + Bigrams.
}

#Note on hyperparameter C:
#C parameter must only be used when overfits occurs, because Regularization adds a penalty term 
# to this cost function, so essentially it changes the objective function to minimize and the problem becomes different from the one without a penalty term.
# C is the inverse regularization, 1 as default, very low values cause coefficients to approach zero and model can underfit, high values cause the opposite.
# C must be used when model is overfitting, adding low C values. With underfitting we should use more high C values.
# Regularization (C): Inverse of regularization strength.
    # C = 1.0 is default.
    # Smaller values (e.g., 0.01) -> Stronger regularization (prevents overfitting). (high variance, memorizing noise).
    # Larger values (e.g., 100) -> Weaker regularization (trusts training data more). (high bias, too simple to capture the signal)

# Apply parameters to the pipeline
cp.set_params(**clsfParams)

# --- 6. Training & Prediction ---

print("\nTraining the model...")
cp.fit(X_train, y_train)

print("Predicting on test set...")
y_pred = cp.predict(X_test)

# --- 7. Evaluation ---

print('\n--- Classification Report ---')
print(classification_report(y_test, y_pred))

print('--- Accuracy Score ---')
print(f"{accuracy_score(y_test, y_pred):.4f}")

# Custom Overfitting Check
# Compares Train Accuracy vs Test Accuracy
check_overfitting(cp, X_train, y_train, X_test, y_test)

# --- 8. Model Persistence (Export) ---
import joblib
import os

model_filename = 'models/bow_logit_v1.joblib'
joblib.dump(cp, model_filename)
print(f"\n💾 Model saved successfully at: {model_filename}")