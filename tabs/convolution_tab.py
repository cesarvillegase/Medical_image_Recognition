import customtkinter as ctk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np 

from logic.get_data import get_training_data
from logic.cnn import train_cnn_model, test_cnn_model

class ConvolutionalTab(ctk.CTkFrame):
    def __init__(self, master):
        self.master = master
        
        # Preprocess Step in the Dataset
        
        self.title_label = ctk.CTkLabel(self.master, text="Convolutional Neural Network", 
                                        anchor="w", font=("bold", 22))
        self.title_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        
        self.train_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                           text="Train Network", font=("normal", 14), command=self.train_network, width=180)
        self.train_button.place(relx=0.015, rely=0.20, relwidth=0.15, relheight=0.05)
        
        self.test_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                          text="Test Network", font=("normal", 14), command=self.test_network, width=180)
        self.test_button.place(relx=0.015, rely=0.30, relwidth=0.15, relheight=0.05)
        
    def train_network(self):
        x_train, y_train = get_training_data(r'input\chest_xray\train')
        x_val, y_val = get_training_data(r'input\chest_xray\val')

        trained_model = train_cnn_model(x_train, y_train, x_val, y_val)
        self.trained_model = trained_model  # Store the trained model for future use

    def test_network(self):
        if hasattr(self, 'trained_model'):  # Check if the model has been trained
            x_test, y_test = get_training_data(r'input\chest_xray\test')
            test_cnn_model(self.trained_model, x_test, y_test)
        else:
            print("Please train the network first.")
