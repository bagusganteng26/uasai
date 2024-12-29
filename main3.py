import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Contoh data sekuensial (teks)
texts = [
    "I love this product, it is amazing!",
    "This is the worst thing I have ever bought.",
    "Absolutely fantastic! I'm very happy with it.",
    "Terrible, I hate it.",
    "It's okay, not the best but not the worst."
]

labels = [1, 0, 1, 0, 1]  # 1: Positive, 0: Negative (label sentimen)

# 2. Tokenisasi dan padding data teks
vocab_size = 1000  # Jumlah maksimum kata dalam tokenizer
max_sequence_len = 10  # Panjang maksimum sekuens

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# 3. Membuat model RNN dengan LSTM
model = Sequential([
    # Embedding layer untuk representasi kata
    Embedding(input_dim=vocab_size, output_dim=16, input_length=max_sequence_len),

    # LSTM layer sebagai inti RNN
    LSTM(32, return_sequences=False),  # return_sequences=False karena hanya butuh output terakhir

    # Dense layer untuk klasifikasi
    Dense(16, activation='relu'),  # Layer tersembunyi
    Dense(1, activation='sigmoid')  # Output layer (sigmoid untuk biner)
])

# 4. Kompilasi model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Latih model
padded_sequences = np.array(padded_sequences)
labels = np.array(labels)

model.fit(padded_sequences, labels, epochs=10, batch_size=2)

# 6. Prediksi
new_texts = ["I absolutely love it!", "This is so bad."]
new_sequences = tokenizer.texts_to_sequences(new_texts)
new_padded_sequences = pad_sequences(new_sequences, maxlen=max_sequence_len, padding='post')

predictions = model.predict(new_padded_sequences)
for i, text in enumerate(new_texts):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {'Positive' if predictions[i] > 0.5 else 'Negative'}")