import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Set paths
base_dir = './'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')
validation_csv_path = os.path.join(base_dir, 'validation.csv')
image_size = (128, 128)

# Load datasets
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
validation_df = pd.read_csv(validation_csv_path)

# Function to load images
def load_images(dataframe, folder):
    images = []
    for img_id in dataframe['image_id']:
        img_path = os.path.join(base_dir, folder, f'{img_id}.png')
        img = Image.open(img_path).convert('RGB').resize(image_size)  # Convert image to RGB
        images.append(np.array(img))
    return np.array(images)
# Load and preprocess images
train_images = load_images(train_df, 'train')
validation_images = load_images(validation_df, 'validation')
train_images = train_images / 255.0
validation_images = validation_images / 255.0

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['label'])
validation_labels = label_encoder.transform(validation_df['label'])

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(train_labels)), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(validation_images, validation_labels))

# Load test images and make predictions
test_images = load_images(test_df, 'test')
test_images = test_images / 255.0
predictions = model.predict(test_images)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

# Create submission file
submission_df = pd.DataFrame({'image_id': test_df['image_id'], 'label': predicted_labels})
submission_df.to_csv(os.path.join(base_dir, 'submission2.csv'), index=False)

print("Model training complete and submission file created.")