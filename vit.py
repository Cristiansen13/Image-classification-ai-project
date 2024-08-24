import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Set paths
base_dir = './'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')
validation_csv_path = os.path.join(base_dir, 'validation.csv')

# Load datasets
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
validation_df = pd.read_csv(validation_csv_path)

# Function to load and preprocess images
def load_images(dataframe, folder):
    images = []
    for img_id in dataframe['image_id']:
        img_path = os.path.join(base_dir, folder, f'{img_id}.png')
        img = Image.open(img_path).convert('RGB').resize((128, 128))
        images.append(np.array(img))
    return np.array(images) / 255.0

# Load and preprocess images
train_images = load_images(train_df, 'train')
validation_images = load_images(validation_df, 'validation')

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['label'])
validation_labels = label_encoder.transform(validation_df['label'])

# Custom layer for creating patches from images
class Patches(layers.Layer):
    def _init_(self, patch_size, **kwargs):
        super(Patches, self)._init_(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# Encoder for projecting patches to a higher-dimensional space
class PatchEncoder(layers.Layer):
    def _init_(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self)._init_(**kwargs)
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# Building the Vision Transformer model
def create_vit_classifier():
    inputs = layers.Input(shape=(128, 128, 3))
    patches = Patches(patch_size=16)(inputs)
    encoded_patches = PatchEncoder(num_patches=(128 // 16) ** 2, projection_dim=128)(patches)
    for _ in range(8):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=2048, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(units=128, activation=tf.nn.gelu)(x3)
        encoded_patches = layers.Add()([x3, x2])
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    outputs = layers.Dense(len(np.unique(train_labels)), activation="softmax")(representation)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Build, compile, and train the model
model = create_vit_classifier()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=30, validation_data=(validation_images, validation_labels))

# Load test images, predict and generate submission file
test_images = load_images(test_df, 'test')
predictions = model.predict(test_images)
predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
submission_df = pd.DataFrame({'image_id': test_df['image_id'], 'label': predicted_labels})
submission_df.to_csv(os.path.join(base_dir, 'new_last_submission.csv'), index=False)

print("Model training complete and submission file created.")