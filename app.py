# app.py
from flask import Flask, render_template, request
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import nltk

app = Flask(__name__)

nltk.download('stopwords')
stopwords_list = set(nltk.corpus.stopwords.words('english'))

# Load model and tokenizer
model = tf.keras.models.load_model("sentiment_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = joblib.load(f)

max_len = 200

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords_list])
    return text

def predict_sentiment(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0]

    confidence = prediction.max()
    label_index = prediction.argmax()

    if label_index == 0:
        sentiment = "Negative"
    elif label_index == 1:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"

    return sentiment, round(float(confidence) * 100, 2)




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['GET','POST'])
def analyze():
    sentiment = None
    confidence = None
    if request.method == 'POST':
        user_input = request.form.get('text-input', '')
        sentiment, confidence = predict_sentiment(user_input)
    return render_template('analyze.html', sentiment=sentiment, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
