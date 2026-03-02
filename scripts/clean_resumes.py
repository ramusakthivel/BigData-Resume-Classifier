import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load Data directly from your CSV
print("Loading dataset...")
df = pd.read_csv('../data/resume_data.csv')

# 2. Basic Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+\s*', ' ', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)     # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
    return text

print("Cleaning resumes (this may take a minute)...")
df['cleaned_resume'] = df['Resume_str'].apply(clean_text)

# 3. Label Encoding (Convert 'HR', 'SALES' etc. into numbers 0, 1, 2...)
le = LabelEncoder()
df['category_id'] = le.fit_transform(df['Category'])

# 4. Save for Phase 3 (Deep Learning)
df.to_pickle('/home/ramusakthivel/resume_project/cleaned_data.pkl')
print(f"Success! Processed {len(df)} resumes.")
print("Sample Categories:", df['Category'].unique()[:5])
