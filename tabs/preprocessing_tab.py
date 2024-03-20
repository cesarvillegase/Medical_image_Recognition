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
        
    def load_data(self):
        data_dir = r'input\chest_xray\train'
        images, labels = get_training_data(data_dir)
        
        # Store the loaded data 
        self.images = images
        self.labels = labels
        
        # Display the first images 
        self.display_iamges(images[:5], labels[:5])
        
    def display_images(self, images, labels):
        # Display the images and labels in the canvas
        self.canvas.delete("all")  # Clear previous images
        
        # Display images
        for i, (image, label) in enumerate(zip(images, labels)):
            # Convert numpy array to ImageTk format
            image = Image.fromarray(image)
            image_tk = ImageTk.PhotoImage(image)
            
            # Display image on canvas
            x_offset = i * 100
            self.canvas.create_image(x_offset, 0, anchor="nw", image=image_tk)
            
            # Display label below the image
            self.canvas.create_text(x_offset, 150, anchor="nw", text=f"Label: {label}")
            
            # Keep a reference to avoid garbage collection
            self.canvas.image = image_tk