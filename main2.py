from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

# Membuat model Sequential
model = Sequential()

# Menambahkan Convolutional Layer
model.add(Conv2D(
    filters=32,               # Jumlah filter/kernels yang akan diterapkan
    kernel_size=(3, 3),       # Ukuran kernel (3x3)
    activation='relu',        # Fungsi aktivasi yang digunakan
    input_shape=(64, 64, 3)   # Ukuran input (gambar 64x64 dengan 3 channel RGB)
))

# Menambahkan lapisan Flatten untuk meratakan output ke dalam vektor 1D
model.add(Flatten())

# Menambahkan Fully Connected Layer untuk klasifikasi
model.add(Dense(
    units=1,                  # Jumlah neuron (1 neuron untuk klasifikasi biner)
    activation='sigmoid'      # Fungsi aktivasi sigmoid untuk probabilitas biner
))

# Menampilkan ringkasan model
model.summary()
