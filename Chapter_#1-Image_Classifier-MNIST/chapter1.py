import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras import layers, Sequential
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy


# Download the dataset
(train_dataset, test_dataset), info = tfds.load('mnist', split=['train', 'test'], with_info=True, as_supervised=True)

# Create the model 
model = Sequential()
model.add(layers.Input(shape=(28, 28, 1), name='Input_Layer'))
model.add(layers.Rescaling(scale=1./255, name='Rescaling_Layer'))
model.add(layers.Flatten(name='Flattening_Layer'))
model.add(layers.Dense(units=64, activation='relu', name='Hidden_Layer_1'))
model.add(layers.Dense(units=10, name='Output'))

# compile the model
model.compile(
    optimizer='adam', 
    loss=SparseCategoricalCrossentropy(from_logits=True), 
    metrics=SparseCategoricalAccuracy()
)

model.summary()
tf.keras.utils.plot_model(model, 'Chapter_#1-Image_Classifier-MNIST/model.png')

# Train the model
model.fit(train_dataset.batch(16), epochs=1)

# Test the model
for x, y in test_dataset.take(1):
    output = model(tf.reshape(x, (1, 28, 28, 1)))
    print('Model Output: ', output)
    print('Label:', y)

# Evaluating the model
model.evaluate(test_dataset.batch(16))
