import json
import re
import os
import joblib
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Text cleansing function
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r'^https?:\/\/.*[\r\n]*', '', txt)
    txt = re.sub(r'@\w+', '', txt)
    txt = re.sub(r'#\d+\b', '', txt)
    txt = re.sub(r'#\d+(\w+)', r'\1', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

# Reading the dataset
df = pd.read_csv('C:\\Users\\pablo\\OneDrive\\Escritorio\\Master\\TFM\\docker\\data\\df_inputed.csv', index_col = 0)
df = df[['message', 'label']]
seed=7764

# Applying text cleansing
df['message'] = df['message'].apply(lambda x: clean_text(x))

# Preparing data
y = df.pop('label')
X = df['message']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Model pipeline
pipe = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('svm', SVC(kernel='linear', probability=True))
])

# Fitting pipeline
pipe.fit(X_train, y_train)

# Getting predictions
predictions = pipe.predict(X_test)
print(classification_report(y_test, predictions))

# Plot confusion matrix
print(ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test))
plt.show()

# Export model
joblib.dump(pipe, gzip.open('C:\\Users\\pablo\\OneDrive\\Escritorio\\Master\\TFM\\docker\\model\\model_binary.dat.gz', "wb"))