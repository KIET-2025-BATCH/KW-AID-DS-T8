# model.py
import numpy as np
import pandas as pd
import re
import string
import nltk
import joblib
import zipfile
import tensorflow as tf
from nltk.corpus import stopwords
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))

df = pd.read_csv("C:/Users/Charishma Chikkala/Downloads/twitter_sentimenet/twitter_sentimenet/balanced_sentiment_dataset (1).csv", encoding='latin-1', header=None)

df.columns = ["label", "id", "date", "query", "username", "text"]
df = df[["text", "label"]]

# Filter only 0, 2, 4
df = df[df['label'].isin([0, 2, 4])]
df['label'] = df['label'].replace({0: 0, 2: 1, 4: 2})  # remap

# Sample balanced classes
neg = df[df['label'] == 0].sample(20000, random_state=42)
neu = df[df['label'] == 1].sample(20000, random_state=42)
pos = df[df['label'] == 2].sample(20000, random_state=42)

data = pd.concat([neg, neu, pos])

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = " ".join([word for word in text.split() if word not in stopwords_list])
    return text

data['text'] = data['text'].apply(preprocess_text)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=200)
Y = to_categorical(data['label'], num_classes=3)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)

# Model
inputs = Input(shape=(200,))
x = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=50)(inputs)
x = LSTM(64, return_sequences=True)(x)
x = LSTM(64)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
checkpoint = ModelCheckpoint("sentiment_model.h5", save_best_only=True)

model.fit(X_train, Y_train, validation_split=0.1, epochs=5, batch_size=64,
          callbacks=[early_stop, checkpoint])

# Save
model.save("sentiment_model.h5")
with open("tokenizer.pkl", "wb") as f:
    joblib.dump(tokenizer, f)

print("✅ Model and tokenizer saved!")
