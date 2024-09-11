import pandas as pd
from ms import model

# Model prediction
def predict(X, model):
    prediction = model.precit(X)[0]
    return prediction

# Loads the DF and predicts the result, returning another JSON object with the request status, predicted label and value
def get_model_response(json_data):
    X = pd.DataFrame.from_dict(json_data)
    prediction = predict(X, model)
    if prediction == 0:
        label = "Sano"
    elif prediction == 1:
        label = "Ansiedad"
    elif prediction == 2:
        label = "Depresión"
    else:
        label = "TCA"
    return {
        'status': 200,
        'label': label,
        'prediction': int(prediction)
    }