import tensorflow as tf
from tensorflow.keras import layers, models

# Membuat model CNN
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()
    
    # Lapisan konvolusi pertama
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Lapisan konvolusi kedua
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Lapisan konvolusi ketiga
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Lapisan flatten dan fully connected
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Contoh penggunaan
input_shape = (64, 64, 3)  # Misalnya, gambar berukuran 64x64 dengan 3 saluran warna
num_classes = 10            # Jumlah kelas dalam dataset
cnn_model = create_cnn_model(input_shape, num_classes)
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Penjelasan Arsitektur CNN:
# Lapisan Konvolusi: Tiga lapisan konvolusi digunakan untuk mengekstrak fitur dari gambar. Setiap lapisan memiliki filter dengan ukuran (3, 3).
# Max Pooling: Setelah setiap lapisan konvolusi, lapisan pooling digunakan untuk mengurangi dimensi fitur dan mengurangi risiko overfitting.
# Dense Layer: Setelah lapisan flatten, lapisan dense digunakan untuk klasifikasi akhir, di mana outputnya menggunakan fungsi aktivasi softmax untuk multi-class classification.