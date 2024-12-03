import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# Set paths to your training and validation directories
train_directory = 'C:\\Users\\Test\\PycharmProjects\\pythonProject2\\dataset-resized'
validation_directory = 'C:\\Users\\Test\\PycharmProjects\\pythonProject2\\dataset-resized'

# Set image parameters
img_height, img_width = 224, 224  # Standard size for pretrained models like MobileNetV2
batch_size = 32

# Create data generators for loading images with augmented training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# Load training and validation data
train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_directory,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

# Load a pretrained MobileNetV2 model without the top layers
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Freeze the base model layers to retain pretrained weights
base_model.trainable = False

# Create the new model on top of the base model
model = keras.Sequential([
    base_model,  # Use MobileNetV2 as feature extractor
    layers.GlobalAveragePooling2D(),  # Pooling layer to reduce dimensionality
    layers.Dense(128, activation='relu'),  # Fully connected layer for learning specific dataset features
    layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model with the optimizer and loss
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),  # Lowered learning rate for transfer learning
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=20,
    verbose=1
)

# Save the model
model.save('my_model')
