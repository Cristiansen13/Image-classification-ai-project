import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Layer
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Embedding, Add
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

# Set paths
base_dir = './'
train_csv_path = os.path.join(base_dir, 'train.csv')
test_csv_path = os.path.join(base_dir, 'test.csv')
validation_csv_path = os.path.join(base_dir, 'validation.csv')

# Load datasets
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
validation_df = pd.read_csv(validation_csv_path)

# Function to load images
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
class Patches(Layer):
    def __init__(self, patch_size=16, **kwargs):
        super(Patches, self).__init__(**kwargs)
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
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

# CNN Model
def create_cnn_model(learning_rate=0.001):
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
    return model

# Vision Transformer Model
def create_vit_model(learning_rate=0.001):
    inputs = Input(shape=(128, 128, 3))
    patches = Patches(patch_size=16)(inputs)
    encoded_patches = PatchEncoder(num_patches=(128 // 16) ** 2, projection_dim=128)(patches)
    for _ in range(8):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = MultiHeadAttention(num_heads=8, key_dim=128, dropout=0.1)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = Dense(units=2048, activation='gelu')(x3)
        x3 = Dropout(0.1)(x3)
        x3 = Dense(units=128, activation='gelu')(x3)
        encoded_patches = Add()([x3, x2])
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    outputs = Dense(len(np.unique(train_labels)), activation="softmax")(representation)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Hyperparameter tuning setup
learning_rates = [0.1, 0.01, 0.001, 0.0001]
batch_sizes = [32, 64, 128]
num_epochs = [10, 20, 30]

results_list = []
model_type = 'cnn'  # Choose 'cnn' or 'vit' as needed

for lr in learning_rates:
    for batch in batch_sizes:
        for epoch in num_epochs:
            if model_type == 'cnn':
                model = create_cnn_model(learning_rate=lr)
            else:
                model = create_vit_model(learning_rate=lr)
            model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(train_images, train_labels, epochs=epoch, batch_size=batch, validation_data=(validation_images, validation_labels), verbose=0)

            # Logging results
            best_val_acc = max(history.history['val_accuracy'])
            final_train_acc = history.history['accuracy'][-1]
            results_list.append({
                'Learning Rate': lr,
                'Batch Size': batch,
                'Epochs': epoch,
                'Training Accuracy': final_train_acc,
                'Validation Accuracy': best_val_acc
            })

# Convert results to DataFrame and visualize
results = pd.DataFrame(results_list)
results.to_csv('hyperparameter_tuning_results.csv', index=False)

# Visualizations
sns.lineplot(data=results, x='Learning Rate', y='Validation Accuracy', hue='Epochs', style='Batch Size', markers=True)
plt.xscale('log')
plt.title('Impact of Learning Rate on Validation Accuracy')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Validation Accuracy')
plt.show()

pivot_table = results.pivot_table(values='Validation Accuracy', index='Batch Size', columns='Epochs')
sns.heatmap(pivot_table, annot=True, cmap='viridis')
plt.title('Validation Accuracy for Different Batch Sizes and Epochs')
plt.xlabel('Epochs')
plt.ylabel('Batch Size')
plt.show()

sns.barplot(x='Learning Rate', y='Validation Accuracy', data=results, color='blue', label='Validation Accuracy')
sns.barplot(x='Learning Rate', y='Training Accuracy', data=results, color='lightblue', label='Training Accuracy')
plt.title('Training vs Validation Accuracy for Different Learning Rates')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

sns.scatterplot(data=results, x='Learning Rate', y='Validation Accuracy', size='Epochs', hue='Batch Size', style='Batch Size', sizes=(20, 200), alpha=0.5)
plt.xscale('log')
plt.title('Impact of Learning Rate and Batch Size on Validation Accuracy')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Validation Accuracy')
plt.legend(title='Batch Size/Epochs')
plt.show()
