from flask import Flask, request, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

# Load your model and vectorizer
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Clean the input text
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        email = request.form['email']
        cleaned = clean_text(email)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        result = "Phishing" if prediction == 1 else "Legitimate"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)