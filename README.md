# README: Fashion MNIST Classification with TensorFlow

## Project Overview
This project focuses on building a neural network model using **TensorFlow** and **Keras** to classify images from the **Fashion MNIST dataset**. The dataset consists of 70,000 grayscale images of 10 different types of clothing items. The goal is to develop and train a deep learning model to correctly classify these images into their respective categories.

## Dataset
The **Fashion MNIST dataset** includes:
- **60,000 training images**
- **10,000 test images**

Each image is 28x28 pixels, and the corresponding labels represent the clothing category:
1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Model Architecture
The model is a **Sequential Neural Network** that includes the following layers:
1. **Flatten Layer**: Converts each 28x28 image into a 1D vector of 784 elements.
2. **Dense Layer 1**: A fully connected layer with 300 neurons and ReLU activation function.
3. **Dense Layer 2**: A fully connected layer with 100 neurons and ReLU activation function.
4. **Dense Output Layer**: A fully connected output layer with 10 neurons, representing each class, using the softmax activation function for classification.

## Training the Model
The training data is split into training and validation sets, and the model is trained on the normalized pixel values (scaled between 0 and 1) of the Fashion MNIST images. The training process involves optimizing the model's parameters using **categorical cross-entropy** as the loss function and **Adam** optimizer.

## Required Libraries
The following Python libraries are required:
- `tensorflow`
- `keras`
- `numpy`
- `pandas`
- `matplotlib`

## Results
After training, the model is evaluated using accuracy metrics on both the validation and test datasets. You can visualize sample predictions or model performance with `matplotlib`.

## Visualizing Model
You can visualize the architecture of the model using:
```python
keras.utils.plot_model(model)
```

## Conclusion
This project demonstrates how to apply a basic neural network for image classification tasks using the Fashion MNIST dataset.
