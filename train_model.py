import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

nltk.download('stopwords')

# Load dataset
data = pd.read_csv('emails.csv')

# Clean text
def clean_text(text):
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"\\W", " ", text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')