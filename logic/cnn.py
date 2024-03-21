import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def train_cnn_model(x_train, y_train, x_val, y_val):
    # Normalize the input data
    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255

    # Define the CNN model
    height, width = x_train.shape[1:]  # Get height and width from the shape of the patterns
    channels = 1
    num_classes = 2

    model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=1, validation_data=(x_val, y_val))

    # Evaluate the model on the combined patterns
    loss, accuracy = model.evaluate(x_train, y_train)  

    return model, history

def test_cnn_model(model, x_test, y_test):
    # Normalize the test data
    x_test = np.array(x_test) / 255

    # Get predictions for the pattern
    predictions = model.predict(x_test)
    print("The first 5 predictions")
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    print("Predicted class:", predicted_class)
    
    # Compare with the true label
    print("True label:", y_test)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, accuracy

def plot_loss(loss_values):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.plot(range(1, len(loss_values) + 1), loss_values, color='orange', label='MSE')
        plt.title("Training for the Auto encoder")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        print(loss_values[-1])