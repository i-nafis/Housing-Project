# Housing Price Prediction

This project is a simple machine learning model built using TensorFlow to predict house prices based on input data. The model is trained on a dataset that simulates house sizes and corresponding prices and demonstrates fundamental concepts in machine learning such as data preprocessing, model building, training, and visualization.

## Project Overview

The goal of this project is to predict house prices based on a single feature: the size of the house. The model is a linear regression model built with TensorFlow, which learns the relationship between the input (house size) and the output (price) by minimizing mean squared error.

## Features

- **Data Preprocessing**: Normalize input data to ensure better model performance.
- **Model Architecture**: A simple linear regression model with one dense layer to capture the linear relationship.
- **Training and Evaluation**: The model is trained using the SGD optimizer with a low learning rate for better convergence.
- **Visualization**: Plots actual data points and the model’s predictions to show how well the model has learned the trend.

## Dataset

The dataset used for training is a synthetic dataset that simulates house sizes and prices. It is defined as follows:

| House Size | House Price |
|------------|-------------|
| 1.0        | 1.0         |
| 2.0        | 1.5         |
| 3.0        | 2.0         |
| 4.0        | 2.5         |
| 5.0        | 3.0         |
| 6.0        | 3.5         |

This data has a linear relationship where each increase in house size by 1 unit corresponds to an increase in price by 0.5 units.

## Model

The model is a simple neural network created using TensorFlow's Sequential API, with one dense layer:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),  # Input layer
    tf.keras.layers.Dense(1)            # Dense layer for linear regression
])
```
The model uses Mean Squared Error (MSE) as the loss function and Stochastic Gradient Descent (SGD) as the optimizer with a low learning rate to better capture the linear relationship.
