import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator




# Load the TensorFlow model
converter = tf.lite.TFLiteConverter.from_saved_model('C:\\Users\\Test\\PycharmProjects\\pythonProject2\\my_model')

# Optional: Optimize the model for better performance
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Use default optimizations
# converter.target_spec.supported_types = [tf.float16]  # Optional float16 quantization

# Convert the model to TensorFlow Lite format
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)