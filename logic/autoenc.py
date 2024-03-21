import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class AutoEncoder:
    def __init__(self):
        self.input_neurons = None
        self.hidden_neurons = None
        self.output_neurons = None
        self.epoch = 0
        self.loss = []

    @staticmethod
    def sigmoid(x):
        return float(1) / (float(1) + np.exp(-x))

    @staticmethod
    def sigmoid_dev(x):
        return x * (float(1) - x)

    def train(self, x_train, alpha, momentum, epoch_max):
        inputs = np.array(x_train) / 255
        inputs = inputs.flatten()
        expected_output = inputs

        self.input_neurons = inputs.shape[0]
        self.hidden_neurons = 3
        self.output_neurons = self.input_neurons

        weights_input = 2 * np.random.random((self.input_neurons, self.hidden_neurons)) - 1
        weights_output = 2 * np.random.random((self.hidden_neurons, self.output_neurons)) - 1

        w_old_input = np.zeros_like(weights_input)
        w_new_input = np.zeros_like(weights_input)

        w_old_output = np.zeros_like(weights_output)
        w_new_output = np.zeros_like(weights_output)

        mse = float(2)
        prev_mse = float(0)

        while (self.epoch < epoch_max) and abs(mse - prev_mse) > 0.00001:
            prev_mse = mse

            hidden_lyr_input = np.dot(inputs, weights_input)
            hidden_lyr_output = self.sigmoid(hidden_lyr_input)

            output_lyr_input = np.dot(hidden_lyr_output, weights_output)
            output = self.sigmoid(output_lyr_input)

            output_error = expected_output - output
            mse = np.mean(output_error ** 2)
            self.loss.append(mse)
            gradient = output_error * self.sigmoid_dev(output)

            output_lyr_delta = gradient

            hidden_lyr_error = output_lyr_delta.dot(weights_output.T)
            hidden_lyr_delta = hidden_lyr_error * self.sigmoid_dev(hidden_lyr_output)

            w_new_output = weights_output + alpha * np.outer(hidden_lyr_output, output_lyr_delta) + momentum * (
                    weights_output - w_old_output)
            w_old_output = weights_output.copy()
            weights_output = w_new_output

            w_new_input = weights_input + alpha * np.outer(inputs, hidden_lyr_delta) + momentum * (
                    weights_input - w_old_input)
            w_old_input = weights_input.copy()
            weights_input = w_new_input

            if self.epoch % 100 == 0:
                print(f"Epoch: {self.epoch} Error: {mse}")

            self.epoch += 1

        latent_space: float = self.sigmoid(np.dot(inputs, weights_input))
        decoded_inputs = self.sigmoid(np.dot(latent_space, weights_output))
        decoded_inputs = (decoded_inputs * 255).astype(int)

        return self.loss, latent_space, decoded_inputs


def plot_loss_ac(loss_values):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        plt.plot(range(1, len(loss_values) + 1), loss_values, color='orange', label='MSE')
        plt.title("Training for the Auto encoder")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("images\_autoenc\image_1.png")
        plt.close()
        # plt.show()

        print(loss_values[-1])


def plot_images_ac(original_image, reconstructed_img):
    """Plot the original, noisy, and reconstructed images."""
    plt.figure(figsize=(8, 4))

    # Reshape and convert the original image to a NumPy array
    original_image_array = np.array(original_image[0])
    original_image_array = original_image_array.astype(np.uint8)  # + 1) / 2 * 255

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image_array)
    plt.title('Original Image')
    plt.axis('off')  # Turn off axes

    # Plot the reconstructed image
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img)
    plt.title('Reconstructed Image')
    plt.axis('off')
    plt.tight_layout()
    