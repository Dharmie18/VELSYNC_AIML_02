import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

print("Loading Horses vs Humans dataset...")

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'horses_or_humans',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

IMG_SIZE = 128

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

# Batch and prefetch
BATCH_SIZE = 32
ds_train = ds_train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary: 0=Horse, 1=Human
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n Model Summary:")
model.summary()

# Train (only 10 epochs)
print("\n Training...")
history = model.fit(
    ds_train,
    epochs=10,
    validation_data=ds_test,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# Show sample predictions
class_names = ["Horse", "Human"]
for images, labels in ds_test.take(1):
    preds = model.predict(images[:5])
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        pred_class = class_names[int(preds[i][0] > 0.5)]
        true_class = class_names[labels[i].numpy()]
        color = 'green' if pred_class == true_class else 'red'
        plt.title(f"Pred: {pred_class}\nTrue: {true_class}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("sample_predictions.png")
    plt.show()
    break

print("\n Task 2 Complete! Horses vs Humans classifier trained.")