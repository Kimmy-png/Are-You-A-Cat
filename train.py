import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Rescaling, RandomFlip, RandomRotation
import numpy as np
import os
import matplotlib.pyplot as plt

print("TensorFlow Versi:", tf.__version__)


PATH_KUCING = 'cat_dataset'
PATH_MANUSIA = 'human_dataset'

BATCH_SIZE = 64 
IMG_SIZE = (128, 128)
JUMLAH_SAMPEL = 4000 


print("Menyiapkan daftar file gambar...")


print("Mencari file gambar kucing di semua subfolder...")
cat_files = []
image_extensions = ('.jpg', '.jpeg', '.png')

for root, dirs, files in os.walk(PATH_KUCING):
    for filename in files:
        if filename.lower().endswith(image_extensions):
            cat_files.append(os.path.join(root, filename))

cat_files = cat_files[:JUMLAH_SAMPEL]

human_files = [
    os.path.join(PATH_MANUSIA, fname) 
    for fname in os.listdir(PATH_MANUSIA) 
    if fname.lower().endswith(image_extensions)
][:JUMLAH_SAMPEL]

print(f"Jumlah gambar kucing yang ditemukan dan digunakan: {len(cat_files)}")
print(f"Jumlah gambar manusia yang digunakan: {len(human_files)}")

cat_labels = [1] * len(cat_files)
human_labels = [0] * len(human_files)

all_files = cat_files + human_files
all_labels = cat_labels + human_labels

path_ds = tf.data.Dataset.from_tensor_slices(all_files)
label_ds = tf.data.Dataset.from_tensor_slices(all_labels)

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    return image, label


full_ds = tf.data.Dataset.zip((path_ds, label_ds))
full_ds = full_ds.shuffle(buffer_size=len(all_files)).map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

DATASET_SIZE = len(all_files)
train_size = int(0.8 * DATASET_SIZE)
val_size = int(0.2 * DATASET_SIZE)

train_ds = full_ds.take(train_size)
val_ds = full_ds.skip(train_size)

train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

class_names = ['Manusia', 'Kucing']

data_augmentation = Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
])

model = Sequential([
    
    Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    data_augmentation,
    
    # Blok Konvolusi 1
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 2
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Blok Konvolusi 3
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Flatten layer untuk mengubah data menjadi 1 dimensi
    Flatten(),
    
    # Dense layer untuk learning
    Dense(128, activation='relu'),
    
    # Dropout akan mematikan 50% neuron secara acak selama training
    # untuk mencegah overfitting.
    tf.keras.layers.Dropout(0.5),

    # Output layer dengan 2 neuron (Manusia, Kucing) dan aktivasi softmax
    Dense(len(class_names), activation='softmax') 
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan ringkasan arsitektur model
model.summary()

print("\nMemulai proses training model...")
EPOCHS = 10 
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)
print("Training selesai.")

print("\nMenyimpan model ke file 'model.keras'...")

model.save('model.keras')
print("Model berhasil disimpan!")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


def prediksi_gambar(image_path):
    """Fungsi untuk memprediksi sebuah gambar dan memberikan persentase kemiripan."""
    # Muat dan proses gambar
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Buat batch

    # Lakukan prediksi
    predictions = model.predict(img_array)
    skor = predictions[0]

    # Dapatkan nama kelas dan persentase kemiripan
    persentase_kucing = skor[1] * 100
    
    # Tampilkan gambar dan hasil
    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Kemiripan dengan Kucing: {persentase_kucing:.2f}%")
    plt.axis("off")
    plt.show()


print("\nMelakukan uji coba prediksi pada gambar contoh...")
# Coba prediksi salah satu gambar dari dataset kucing
if cat_files:
    contoh_gambar_kucing = cat_files[10] # Ambil gambar ke-10 sebagai contoh
    prediksi_gambar(contoh_gambar_kucing)

# Coba prediksi salah satu gambar dari dataset manusia
if human_files:
    contoh_gambar_manusia = human_files[10] # Ambil gambar ke-10 sebagai contoh
    prediksi_gambar(contoh_gambar_manusia)

