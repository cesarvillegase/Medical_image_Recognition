import os 
import cv2

import customtkinter as ctk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import numpy as np

from logic.get_data import get_training_data

class PreprocessingTab(ctk.CTkFrame):
    def __init__(self, master):
        self.master = master
        
        # Preprocess Step in the Dataset
        
        self.title_label = ctk.CTkLabel(self.master, text="Preprocessing", anchor="w", 
                                        font=("bold", 22))
        self.title_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        
        self.load_dataset_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                                 text="Load Dataset", font=("normal", 14), command=self.load_data, width=180)
        self.load_dataset_button.place(relx=0.015, rely=0.20, relwidth=0.15, relheight=0.05)
        
        # Create canvas to display images
        self.image_1 = Image.new("RGB", (240, 240), "white")
        self.image_2 = Image.new("RGB", (240, 240), "white")
        self.image_3 = Image.new("RGB", (240, 240), "white")
        
        self.image_label_1 = ctk.CTkLabel(self.master, text="Plot")
        self.image_label_1.place(relx=0.3, rely=0.3)
        self.image_widget_1 = ctk.CTkLabel(self.master, image=ImageTk.PhotoImage(self.image_1), text="")
        self.image_widget_1.place(relx=0.3, rely=0.35)
        
        self.image_label_2 = ctk.CTkLabel(self.master, text="Dataset")
        self.image_label_2.place(relx=0.5, rely=0.3)
        self.image_widget_2 = ctk.CTkLabel(self.master, image=ImageTk.PhotoImage(self.image_2), text="")
        self.image_widget_2.place(relx=0.5, rely=0.35)
        
        self.image_label_3 = ctk.CTkLabel(self.master, text="Dataset")
        self.image_label_3.place(relx=0.7, rely=0.3)
        self.image_widget_3 = ctk.CTkLabel(self.master, image=ImageTk.PhotoImage(self.image_3), text="")
        self.image_widget_3.place(relx=0.7, rely=0.35)
        
    def load_data(self):
        data_dir = r'input\chest_xray\train'
        x_train, y_train = get_training_data(data_dir)
        
        labels = ['PNEUMONIA', 'NORMAL']
        
        # Store the loaded data 
        self.x_train = x_train
        self.y_train = y_train
        
        # Plot label distribution for training data
        self.plot_label_distribution(self.y_train, 'Training Label Distribution')
        
        # Display the first images 
        self.plot_images(self.x_train, self.y_train, labels)
        
    def plot_label_distribution(self, labels, title):
        unique_labels, counts = np.unique(labels, return_counts=True)
        colors = ['#ffb703', '#003f88']  # Specify custom colors for each label
        plt.bar(unique_labels, counts, tick_label=unique_labels, color=colors)
        plt.title(title)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.savefig("images\pre_processing\image_1.png")
        plt.close()
        
        image_1 = Image.open("images\pre_processing\image_1.png").resize((240, 240))
        image_1_tk = ImageTk.PhotoImage(image_1)    
        self.image_widget_1.configure(image=image_1_tk)

        
    def plot_images(self, x_train, y_train, labels):
        # Plot the first image
        plt.figure(figsize=(5, 5))
        plt.imshow(x_train[0], cmap='gray')
        plt.title(labels[y_train[0]])  # Get the corresponding label from the labels list
        plt.savefig("images\pre_processing\image_2.png")
        plt.close()

        # Plot the last image
        plt.figure(figsize=(5, 5))
        plt.imshow(x_train[-1], cmap='gray')
        plt.title(labels[y_train[-1]])  # Get the corresponding label from the labels list
        plt.savefig("images\pre_processing\image_3.png")
        plt.close()
        
        image_2 = Image.open("images\pre_processing\image_2.png").resize((240, 240))
        image_3 = Image.open("images\pre_processing\image_3.png").resize((240, 240))
        
        image_2_tk = ImageTk.PhotoImage(image_2)
        image_3_tk = ImageTk.PhotoImage(image_3)
        
        self.image_widget_2.configure(image=image_2_tk)
        self.image_widget_3.configure(image=image_3_tk)
