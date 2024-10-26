import tensorflow as tf

def create_house_model(xs, ys):
    """
    Create, compile, and train a simple neural network model.
    """
    # Define the model using an explicit Input layer
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),         # Explicit Input layer
        tf.keras.layers.Dense(1)                   # Single dense layer for linear regression
    ])

    # Compile the model with SGD optimizer and a lower learning rate for better convergence
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

    # Train the model for a larger number of epochs
    model.fit(xs, ys, epochs=500, verbose=0)

    return model

