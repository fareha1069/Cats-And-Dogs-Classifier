
# Cat vs Dog Image Classifier using CNN

This repository contains a deep learning model built using TensorFlow and Keras to classify images into two categories: **Cat** and **Dog**. The model uses a Convolutional Neural Network (CNN) architecture, which is trained on a dataset of cat and dog images to predict the class of an unseen image.

## Table of Contents

* [Introduction](#introduction)
* [Setup and Installation](#setup-and-installation)
* [Dataset](#dataset)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Training the Model](#training-the-model)
* [Evaluation and Results](#evaluation-and-results)

## Introduction

This project uses a Convolutional Neural Network (CNN) to classify images of cats and dogs. The model is trained using a dataset that contains labeled images of both cats and dogs. After preprocessing and augmenting the data, the model is trained to extract relevant features from the images, such as edges, textures, and shapes. These features are then used to predict whether an image contains a cat or a dog.

## Setup and Installation

To run this project, you need to have the following libraries installed:

* **TensorFlow**: For building and training the CNN model.
* **Keras**: For high-level neural networks API.
* **NumPy**: For numerical operations.
* **OpenCV**: For image processing.
* **Matplotlib**: For visualizing results.

You can install the required libraries using pip.

## Dataset

The dataset consists of images of cats and dogs. It is split into two sets:

* **Training Set**: Used for training the model.
* **Test Set**: Used for evaluating the model's performance.

Images are stored in separate directories based on their class (e.g., "cats" and "dogs").

## Usage

### 1. Data Preprocessing

The images are loaded in batches using the `image_dataset_from_directory` function from Keras, which efficiently handles large datasets by loading data in batches. The images are resized to 256x256 and normalized to the range [0, 1].

Corrupted images are removed from the dataset during preprocessing to avoid errors during training.

### 2. Model Architecture

The CNN model architecture consists of:

1. **Conv2D Layers**: These layers apply filters to the images to extract features such as edges, textures, and patterns.
2. **Batch Normalization**: This normalizes the outputs of the convolution layers to help with faster convergence during training.
3. **MaxPooling2D Layers**: These layers reduce the spatial dimensions of the images to retain the most important features while reducing the computational load.
4. **Flatten Layer**: This layer converts the 2D feature maps into 1D vectors for the fully connected layers.
5. **Dense Layers**: These fully connected layers learn combinations of features to classify the images into one of the two categories (cat or dog).
6. **Dropout Layers**: Dropout is applied to prevent overfitting during training.

The final output layer uses the **sigmoid activation function** to output a probability between 0 and 1, where:

* A value close to 0 indicates "Cat."
* A value close to 1 indicates "Dog."

### 3. Training the Model

The model is compiled using the Adam optimizer and binary cross-entropy loss function. The training process runs for a specified number of epochs, during which the model learns to classify the images.

### 4. Saving and Loading the Model

After training, the model is saved to disk and can be loaded for future predictions.

### 5. Making Predictions

To classify a new image, the model is loaded, and the image is preprocessed to match the input dimensions. The model then predicts the class (cat or dog) for the given image.

### 6. Visualizing Training and Validation Results

The training history is plotted to visualize the model's performance over time, including training and validation accuracy and loss.

## Model Architecture

The CNN model consists of the following layers:

1. **Conv2D Layers**: For feature extraction.
2. **BatchNormalization**: For stabilizing training.
3. **MaxPooling2D**: For dimensionality reduction.
4. **Flatten**: For converting 2D feature maps into 1D vectors.
5. **Dense Layers**: For learning combinations of features.
6. **Dropout**: For preventing overfitting.
7. **Output Layer**: A sigmoid activation for binary classification.

## Training the Model

The model is trained using the Adam optimizer with a learning rate of 0.0003 and binary cross-entropy loss. It is trained for a specified number of epochs, and validation accuracy is monitored after each epoch.

## Evaluation and Results

The model achieves a high accuracy on the validation data. You can visualize the training and validation accuracy and loss over epochs using matplotlib.

To test the model on new, unseen data, you can load the trained model and pass a new image for prediction. The model will classify the image as either a cat or a dog based on the learned features.
