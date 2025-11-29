import pandas as pd
import numpy as np
import os

# Force CPU to avoid GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from preprocess import prepare_data, split_data
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1️⃣ Load and split data
df = prepare_data("data/sms_spam.csv")
X_train, X_test, y_train, y_test = split_data(df)

# 2️⃣ Tokenize
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
joblib.dump(tokenizer, "models/lstm_tokenizer.joblib")

train_seq = tokenizer.texts_to_sequences(X_train)
test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(train_seq, maxlen=100, padding="post")
X_test_pad = pad_sequences(test_seq, maxlen=100, padding="post")

# 3️⃣ Encode labels
y_train = np.array(y_train.map({"ham":0, "spam":1}))
y_test = np.array(y_test.map({"ham":0, "spam":1}))

# 4️⃣ Build BiLSTM
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# 5️⃣ Train
early = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
model.fit(X_train_pad, y_train, epochs=5, batch_size=64,
          validation_split=0.1, callbacks=[early])

# 6️⃣ Evaluate
preds = (model.predict(X_test_pad) > 0.5).astype("int32")
acc = accuracy_score(y_test, preds)
print(f"✅ BiLSTM Accuracy: {acc:.4f}")
print(classification_report(y_test, preds))

model.save("models/bilstm_model.h5")
print("✅ BiLSTM model saved to models/bilstm_model.h5")
