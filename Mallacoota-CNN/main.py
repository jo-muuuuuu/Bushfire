import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load Data
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = image.load_img(img_path, target_size=(512, 512), color_mode="rgb")
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(label)
    return np.array(images), np.array(labels)

height, width = 512, 512

NDVI_burned, labels_burned_1 = load_images_from_folder("TF_DATA/NDVI/Burned", 1)
NDVI_unburned, labels_unburned_1 = load_images_from_folder("TF_DATA/NDVI/Unburned", 0)

NDMI_burned, labels_burned_2 = load_images_from_folder("TF_DATA/NDMI/Burned", 1)
NDMI_unburned, labels_unburned_2 = load_images_from_folder("TF_DATA/NDMI/Unburned", 0)

# Verify the number of labels
assert np.array_equal(labels_burned_1, labels_burned_2)
assert np.array_equal(labels_unburned_1, labels_unburned_2)

X_burned = np.concatenate([NDVI_burned, NDMI_burned], axis=-1)
X_unburned = np.concatenate([NDVI_unburned, NDMI_unburned], axis=-1)

X = np.vstack([X_burned, X_unburned])
y = np.hstack([labels_burned_1, labels_unburned_1])

X = X / 255.0

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(height, width, 6)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 模型训练
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
