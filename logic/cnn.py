import numpy as np
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
    model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

    return model

def test_cnn_model(model, x_test, y_test):
    # Normalize the test data
    x_test = np.array(x_test) / 255

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)