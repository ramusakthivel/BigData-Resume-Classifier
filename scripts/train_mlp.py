import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. Load the cleaned data
print("Loading processed data...")
df = pd.read_pickle('/home/ramusakthivel/resume_project/cleaned_data.pkl')

# 2. Vectorization (Turning text into numbers the AI can understand)
print("Converting text to vectors (TF-IDF)...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned_resume']).toarray()
y = df['category_id'].values

# 3. Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# 4. Build the MLP Architecture
model = Sequential([
    Dense(512, activation='relu', input_shape=(3000,)),
    Dropout(0.3), # Prevents overfitting
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y)), activation='softmax') # Final output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. Train the Model
print("Starting training...")
model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_test, y_test_cat))

# 6. Save the trained brain
model.save('/home/ramusakthivel/resume_project/resume_model.h5')
print("Model saved successfully as resume_model.h5")
