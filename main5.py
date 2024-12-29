import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. Memuat dan Menyiapkan Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data menjadi (jumlah gambar, tinggi, lebar, saluran warna)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# 2. Membangun Arsitektur Model CNN
def create_cnn_model():
    model = models.Sequential()
    
    # Lapisan Konvolusi Pertama
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Lapisan Konvolusi Kedua
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Lapisan Konvolusi Ketiga
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    # Lapisan Flatten dan Fully Connected
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 kelas untuk digit 0-9

    return model

cnn_model = create_cnn_model()

# 3. Menyusun Model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 4. Melatih Model dengan Data Training
history = cnn_model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# 5. Mengevaluasi Performa Model pada Data Testing
test_loss, test_accuracy = cnn_model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Menggambar grafik akurasi
plt.plot(history.history['accuracy'], label='Akurasi Training')
plt.plot(history.history['val_accuracy'], label='Akurasi Validasi')
plt.title('Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()
plt.show()

# Menggambar grafik loss
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validasi')
plt.title('Loss Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Penjelasan Kode:
# Memuat dan Menyiapkan Data:
# Menggunakan mnist.load_data() untuk memuat dataset MNIST.
# Melakukan normalisasi pada data agar nilai pixel berada dalam rentang [0, 1].
# Mengubah bentuk data menjadi (jumlah gambar, tinggi, lebar, saluran warna) untuk CNN.

# Membangun Arsitektur Model CNN:
# Model terdiri dari tiga lapisan konvolusi diikuti oleh lapisan pooling.
# Lapisan flatten untuk mengubah data dari 2D ke 1D, diikuti oleh lapisan dense untuk klasifikasi akhir.

# Menyusun Model:
# Menggunakan optimizer Adam dan fungsi loss sparse categorical crossentropy, yang sesuai untuk masalah klasifikasi multi-kelas.

# Melatih Model:
# Model dilatih dengan data pelatihan selama 5 epoch, dan menggunakan 20% data pelatihan sebagai data validasi.

# Mengevaluasi Model:
# Model diuji menggunakan data pengujian untuk mendapatkan akurasi.
# Grafik akurasi dan loss ditampilkan untuk memantau kinerja model selama pelatihan.