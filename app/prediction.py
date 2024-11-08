import joblib
import pandas as pd

def predict(input_data):
    model = joblib.load('static/trained_model.joblib')
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]