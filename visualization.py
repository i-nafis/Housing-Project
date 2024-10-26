import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(model, xs, ys, new_x):
    """
    Visualize the model's predictions compared to the actual data.
    """
    predictions = model.predict(xs)

    # Convert new_x to a numpy array to avoid the ValueError
    new_x_array = np.array([new_x])
    new_prediction = model.predict(new_x_array)[0]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, 'bo', label='Actual Data')  # Plot actual data points
    plt.plot(xs, predictions, 'r-', label='Model Prediction')  # Plot the model's predictions
    plt.plot(new_x, new_prediction, 'go', label=f'Prediction for {new_x}: {new_prediction[0]:.2f}')

    plt.title("House Price Prediction")
    plt.xlabel("House Size (Arbitrary Units)")
    plt.ylabel("House Price (Arbitrary Units)")
    plt.legend()
    plt.show()
