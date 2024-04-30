
import gradio as gr  # Import Gradio library for creating web interfaces
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import numpy as np  # Import numpy for numerical operations
import os  # Import os for interacting with the operating system
import PIL  # Import PIL for working with images
import tensorflow as tf  # Import TensorFlow for machine learning tasks
from tensorflow import keras  # Import Keras API for building neural networks
from tensorflow.keras import layers # type: ignore
# Import layers module for defining layers in the model
from tensorflow.keras.models import Sequential # type: ignore
# Import Sequential model for linear stack of layers
import pathlib  # Import pathlib for working with file paths

# Download dataset of flower photos and set data directory
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)  # Convert data directory to a Path object

# Get file paths for images of roses
roses = list(data_dir.glob('roses/*'))

# Open and display the first rose image
print(roses[0])
PIL.Image.open(str(roses[0]))

# Define image height and width for resizing
img_height, img_width = 180, 180
batch_size = 32  # Batch size for training

# Create training dataset from directory
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create validation dataset from directory
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names for the dataset
class_names = train_ds.class_names
print(class_names)

# Plot sample images from the training dataset
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# Define model architecture
num_classes = 5
model = Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # Rescale pixel values to [0,1]
    layers.Conv2D(16, 3, padding='same', activation='relu'),  # Convolutional layer with ReLU activation
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Conv2D(32, 3, padding='same', activation='relu'),  # Convolutional layer with ReLU activation
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Conv2D(64, 3, padding='same', activation='relu'),  # Convolutional layer with ReLU activation
    layers.MaxPooling2D(),  # Max pooling layer
    layers.Flatten(),  # Flatten layer to convert 2D matrix data to 1D vector
    layers.Dense(128, activation='relu'),  # Fully connected layer with ReLU activation
    layers.Dense(num_classes, activation='softmax')  # Output layer with softmax activation
])

# Compile the model
model.compile(optimizer='adam',  # Adam optimizer for training
              loss='sparse_categorical_crossentropy',  # Sparse categorical crossentropy loss function
              metrics=['accuracy'])  # Metric to monitor during training

# Train the model
epochs = 10
history = model.fit(
    train_ds,  # Training dataset
    validation_data=val_ds,  # Validation dataset
    epochs=epochs  # Number of epochs
)

# Define a function to predict class probabilities for input image
def predict_image(img):
    img_4d = img.reshape(-1, img_height, img_width, 3)  # Reshape input image to 4D tensor
    prediction = model.predict(img_4d)[0]  # Make prediction using the model
    return {class_names[i]: float(prediction[i]) for i in range(num_classes)}  # Return class probabilities

# Define input and output components for the Gradio interface
image = gr.inputs.Image(shape=(img_height, img_width))  # Input component for image
label = gr.outputs.Label(num_top_classes=5)  # Output component for label

# Create and launch the Gradio interface
gr.Interface(fn=predict_image, inputs=image, outputs=label, interpretation='default').launch(debug=True)
