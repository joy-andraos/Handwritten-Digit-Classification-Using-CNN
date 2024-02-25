import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the CNN model architecture
model = models.Sequential()
# Add convolutional layer with 24 filters, each 3x3, using ReLU activation function
model.add(layers.Conv2D(24, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Add max pooling layer to down-sample the feature maps
model.add(layers.MaxPooling2D((2, 2)))
# Add another convolutional layer with 24 filters, each 3x3, using ReLU activation function
model.add(layers.Conv2D(24, (3, 3), activation='relu'))
# Add max pooling layer to down-sample the feature maps
model.add(layers.MaxPooling2D((2, 2)))
# Add another convolutional layer with 36 filters, each 3x3, using ReLU activation function
model.add(layers.Conv2D(36, (3, 3), activation='relu'))
# Flatten the feature maps to prepare for fully connected layers
model.add(layers.Flatten())
# Add fully connected layer with 900 neurons and ReLU activation function
model.add(layers.Dense(900, activation='relu'))
# Add fully connected layer with 128 neurons and ReLU activation function
model.add(layers.Dense(128, activation='relu'))
# Add output layer with 10 neurons (one for each digit) and softmax activation function
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Reshape data for CNN (add channel dimension)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Train the model with training data, validating on test data for 5 epochs
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the trained model on test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

# Print test loss and accuracy
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

# Make predictions on test images
predictions = model.predict(test_images)

# Generate confusion matrix
conf_matrix = confusion_matrix(test_labels, np.argmax(predictions, axis=1))
print("Confusion Matrix:")
print(conf_matrix)

# Generate classification report
class_report = classification_report(test_labels, np.argmax(predictions, axis=1))
print("\nClassification Report:")
print(class_report)

# Plot the first X test images, their predicted labels, and true labels
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {np.argmax(predictions[i])}, Actual: {test_labels[i]}')
    plt.xticks([])
    plt.yticks([])
    if np.argmax(predictions[i]) == test_labels[i]:
        color = 'blue'
    else:
        color = 'red'
    confidence = np.max(predictions[i]) * 100
    plt.xlabel(f'Confidence: {confidence:.2f}%', color=color)
plt.tight_layout()
plt.show()
