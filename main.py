from model import create_house_model
from visualization import visualize_predictions
import numpy as np

# Define training data
xs = np.array([1.00, 2.00, 3.00, 4.00, 5.00, 6.00], dtype=float)
ys = np.array([1.00, 1.50, 2.00, 2.50, 3.00, 3.50], dtype=float)

# Train model
model = create_house_model(xs, ys)

# Predict and visualize for a new data point
new_x = 7.0
visualize_predictions(model, xs, ys, new_x)
