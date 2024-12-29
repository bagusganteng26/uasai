import tensorflow as tf
from tensorflow.keras import layers, models

# Membuat model RNN
def create_rnn_model(input_length, vocab_size, embedding_dim, num_classes):
    model = models.Sequential()
    
    # Lapisan embedding
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))
    
    # Lapisan LSTM
    model.add(layers.LSTM(128, return_sequences=False))
    
    # Lapisan fully connected
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Contoh penggunaan
input_length = 100       # Panjang input teks
vocab_size = 5000        # Ukuran kosakata
embedding_dim = 128      # Dimensi embedding
num_classes = 5          # Jumlah kelas dalam dataset
rnn_model = create_rnn_model(input_length, vocab_size, embedding_dim, num_classes)
rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Penjelasan Arsitektur RNN:
# Lapisan Embedding: Mengubah kata-kata menjadi vektor dengan dimensi tetap. Ini membantu dalam menangkap makna kata dalam konteks.
# LSTM: Menggunakan Long Short-Term Memory (LSTM) untuk menangkap dependensi jangka panjang dalam data urutan. LSTM lebih baik daripada RNN biasa dalam menangani masalah vanishing gradient.
# Dense Layer: Sama seperti model CNN, lapisan dense digunakan untuk klasifikasi akhir, dengan fungsi aktivasi softmax untuk multi-class classification.