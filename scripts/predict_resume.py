import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# 1. Load the model and the data to recreate the 'categories'
model = load_model('/home/ramusakthivel/resume_project/resume_model.h5')
df = pd.read_pickle('/home/ramusakthivel/resume_project/cleaned_data.pkl')

# 2. Re-initialize the Vectorizer (Must match the training script)
tfidf = TfidfVectorizer(max_features=3000)
tfidf.fit(df['cleaned_resume'])

# 3. Paste a sample resume text here
new_resume = """
Experienced Java Developer with 5 years in building scalable web applications. 
Skilled in Spring Boot, Hibernate, and Microservices. 
Looking for a Senior Software Engineer role in Chennai.
"""

# 4. Clean and Predict
clean_input = new_resume.lower().replace('[^\w\s]', '')
vector = tfidf.transform([clean_input]).toarray()
prediction = model.predict(vector)
category_id = np.argmax(prediction)

# 5. Map the ID back to the name
categories = sorted(df['Category'].unique())
print(f"\n--- AI PREDICTION ---")
print(f"Predicted Category: {categories[category_id]}")
print(f"Confidence: {np.max(prediction)*100:.2f}%")
