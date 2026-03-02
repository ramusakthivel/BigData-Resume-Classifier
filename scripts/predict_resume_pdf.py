import joblib
import numpy as np
import re
from pypdf import PdfReader
from tensorflow.keras.models import load_model

# 1. Extraction and Cleaning (Same as before)
def get_text_from_pdf(pdf_path):
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def clean_text(text):
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# 2. Dynamic Prediction Logic
def predict(pdf_file):
    # Load all 3 required assets using relative paths
    model = load_model('../models/resume_model.h5')
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    label_encoder = joblib.load('../models/label_encoder.pkl') # Loaded dynamically!
    
    # Process the PDF
    raw_text = get_text_from_pdf(pdf_file)
    cleaned_resume = clean_text(raw_text)
    
    # Vectorize
    features = vectorizer.transform([cleaned_resume])
    
    # Predict
    prediction_raw = model.predict(features)
    predicted_class_index = np.argmax(prediction_raw, axis=1)
    
    # Map index back to the original category name
    category_name = label_encoder.inverse_transform(predicted_class_index)[0]
    
    return category_name

# Run it
result = predict("../data/Balakrishnan_Sakthivel_Resume.pdf")
print(f"\n🚀 Predicted Category: {result}")
