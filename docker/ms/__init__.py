import joblib
import os
from flask import Flask

# Initialize App
app = Flask(__name__)

# Load models
model = joblib.load('model/model_binary.dat.gz')