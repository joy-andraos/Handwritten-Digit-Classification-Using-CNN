# Description
Handwritten digit classification is a fundamental problem in the field of computer vision and machine learning. This project utilizes CNNs, a type of deep neural network, to automatically recognize handwritten digits. The MNIST dataset, consisting of 28x28 grayscale images of handwritten digits (0-9), was used for training and testing purposes.

# Architecture of the CNN
Here's the architecture of the CNN that I found most effective for this project:
1) Input Layer: The input layer receives grayscale images of handwritten digits from the MNIST dataset. Each image is 28x28 pixels.
2) Convolutional Layers: The model consists of three convolutional layers, each followed by a ReLU activation function. These layers extract features from the input images using small filters (3x3) and produce feature maps.
3) Pooling Layers: Two max-pooling layers follow the convolutional layers. They reduce the spatial dimensions of the feature maps while retaining the most important information. Max-pooling is performed over 2x2 windows.
4) Flatten Layer: After the convolutional and pooling layers, the feature maps are flattened into a one-dimensional array. This prepares the data for the fully connected layers.
5) Fully Connected Layers: Two dense (fully connected) layers with ReLU activation functions are added. These layers learn high-level features from the flattened data.
6) Output Layer: The output layer consists of 10 neurons, each corresponding to one of the digits (0-9). It uses the softmax activation function to compute the probability distribution over the classes.

# Steps of the training
1) Data Loading: Load the MNIST dataset, which contains grayscale images of handwritten digits along with their labels.
2) Data Preprocessing: Normalize the pixel values of the images to be between 0 and 1 to facilitate training.
3) Model Definition: Define the CNN architecture using the Keras Sequential API. Configure the layers of the model, including convolutional, pooling, flatten, dense, and output layers.
4) Model Compilation: Compile the model by specifying the optimizer (Adam), loss function (sparse categorical cross-entropy), and evaluation metric (accuracy).
5) Data Reshaping: Reshape the training and test data to include a channel dimension required by the convolutional layers.
6) Model Training: Train the CNN model on the training data for a specified number of epochs, validating it on the test data after each epoch.
7) Model Evaluation: Evaluate the trained model's performance on the test data by computing the test loss and accuracy.
8) Prediction: Make predictions on the test images using the trained model to classify the handwritten digits.
9) Visualization: Visualize a subset of test images along with their predicted labels and true labels, highlighting correct and incorrect predictions.

# Results
The trained model achieved an accuracy of 98.83% on the test set, and the confidence of the model can be seen in the vizualization plot attached in the repo.
Additionally, a confusion matrix and a classification report are provided in the code to evaluate the model's performance.
