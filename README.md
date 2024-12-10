# Project-CNN-for-Image-Classification

This repository contains the code and documentation for building a convolutional neural network (CNN) model for image classification, specifically distinguishing between images of cats and dogs.

## 1. Installations
This project was developed using Python, using Jupyter Notebook on Anaconda. with TensorFlow and Keras for building the deep learning model. The relevant Python packages for this project are as follows:
* numpy
* pandas
* tensorflow
* keras
* import matplotlib.pyplot as plt: for data visualization
* ImageDataGenerator : for Image Augmentation and data preprocessing
* from tensorflow.keras.preprocessing.image
* import image
* from tensorflow.keras.preprocessing 

## 2. Project Motivation
In this project, we are going to build up a convolutional neural network (CNN) to be trained on training dataset (including 4000 images of dog and 4000 images of cat), and then be tested on the test dataset including 1000 images of dog and 1000 images of cat (for the sake of evaluation). Finally, you are supposed to input your model with images in "single_prediction" folder in order to classify the image as a dog or cat (making prediction using your model).

The project uses TensorFlow and Keras to construct and train the CNN, which is evaluated on a test set of images.

## 3. File Descriptions
* CNN for Image Classification.ipynb: Python script containing the model architecture, data preprocessing steps, training, and prediction logic.

* dataset/training_set: Directory containing the training set images of dogs and cats.
* dataset/test_set: Directory containing the test set images of dogs and cats.
* dataset/single_prediction: Folder containing images you want to predict as either dogs or cats.
* README.md: This file, which provides an overview of the project.

## 4. Results
The main findings of the code can be found at the post on Medium available [here](https://medium.com/@liuxingbei0726/cnn-model-for-image-classification-specifically-distinguishing-between-images-of-cats-and-dogs-713de863eca5)

