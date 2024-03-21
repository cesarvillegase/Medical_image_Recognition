import numpy as np 
import customtkinter as ctk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

from logic.get_data import get_training_data
from logic.cnn import train_cnn_model, test_cnn_model, plot_loss

class ConvolutionalTab(ctk.CTkFrame):
    def __init__(self, master):
        self.master = master
        
        # Preprocess Step in the Dataset
        
        self.title_label = ctk.CTkLabel(self.master, text="Convolutional Neural Network", 
                                        anchor="w", font=("bold", 22))
        self.title_label.place(relx=0.015, rely=0.05, relwidth=0.35, relheight=0.05)
        
        self.train_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                           text="Train Network", font=("normal", 14), command=self.train_network, width=180)
        self.train_button.place(relx=0.015, rely=0.20, relwidth=0.15, relheight=0.05)
        
        self.test_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                          text="Test Network", font=("normal", 14), command=self.test_network, width=180)
        self.test_button.place(relx=0.215, rely=0.20, relwidth=0.15, relheight=0.05)

        # Load and display image in label 1
        image_1_path = "images/pre_processing/image_2.png"
        self.image_label_1 = ctk.CTkLabel(self.master, text="Plot")
        self.image_label_1.place(relx=0.015, rely=0.3)
        self.image_1 = Image.open(image_1_path)
        self.image_1 = self.image_1.resize((280, 280))
        self.image_1_tk = ImageTk.PhotoImage(self.image_1)
        self.image_widget_1 = ctk.CTkLabel(self.master, image=self.image_1_tk, text="")
        self.image_widget_1.place(relx=0.015, rely=0.35)
        
        # Load and display image in label 2
        image_2_path = "images/pre_processing/image_3.png"
        self.image_label_2 = ctk.CTkLabel(self.master, text="Dataset")
        self.image_label_2.place(relx=0.315, rely=0.3)
        self.image_2 = Image.open(image_2_path)
        self.image_2 = self.image_2.resize((280, 280))
        self.image_2_tk = ImageTk.PhotoImage(self.image_2)
        self.image_widget_2 = ctk.CTkLabel(self.master, image=self.image_2_tk, text="")
        self.image_widget_2.place(relx=0.315, rely=0.35)
        
    def train_network(self):
        x_train, y_train = get_training_data(r'input\chest_xray\train')
        x_val, y_val = get_training_data(r'input\chest_xray\val')

        self.trained_model, self.history = train_cnn_model(x_train, y_train, x_val, y_val)
        
        # Print labels during training
        print("Training Labels:", y_train)
        
        loss_values = self.history.history['loss']
        plot_loss(loss_values)
        
    def test_network(self):
        if hasattr(self, 'trained_model'):  # Check if the model has been trained
            x_test, y_test = get_training_data(r'input\chest_xray\test')
            self.test_loss, self.test_accuracy = test_cnn_model(self.trained_model, x_test, y_test)

            # Print predictions with predicted class and true label in labels
            predictions = self.trained_model.predict(x_test)
            for i in range(5):  # Print first 5 predictions
                predicted_class = np.argmax(predictions[i])
                true_label = y_test[i]
                label_text = f"Predicted: {predicted_class}, True Label: {true_label}"
                label = ctk.CTkLabel(self.master, text=label_text, font=("normal", 12))
                label.place(relx=0.015, rely=0.30 + i * 0.05, relwidth=0.5, relheight=0.05)
        else:
            print("Please train the network first.")