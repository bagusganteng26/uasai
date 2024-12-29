import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Membuat dataset dummy untuk klasifikasi dua kelas
X, y = make_classification(
    n_samples=1000,    # Jumlah sampel
    n_features=20,     # Jumlah fitur
    n_informative=15,  # Fitur yang relevan
    n_classes=2,       # Jumlah kelas (biner)
    random_state=42
)

# 2. Membagi dataset menjadi data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalisasi fitur (penting untuk neural network)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Membangun model neural network
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer dengan 16 neuron
    Dense(1, activation='sigmoid')  # Output layer dengan 1 neuron untuk klasifikasi biner
])

# 5. Mengompilasi model
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Optimizer Adam dengan learning rate 0.001
    loss='binary_crossentropy',          # Fungsi loss untuk klasifikasi biner
    metrics=['accuracy']                 # Metrik evaluasi
)

# 6. Melatih model
history = model.fit(
    X_train, y_train,
    epochs=20,                # Jumlah epoch pelatihan
    batch_size=32,            # Ukuran batch
    validation_split=0.2,     # Memisahkan 20% data latih untuk validasi
    verbose=1                 # Menampilkan log pelatihan
)

# 7. Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy on test data: {accuracy:.2f}")

# 8. Membuat prediksi dan menghitung akurasi
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(f"Final Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
