import customtkinter as ctk
from PIL import Image, ImageTk

from logic.get_data import get_training_data
from logic.autoenc import AutoEncoder, plot_loss_ac

class AutoEncoderTab(ctk.CTkFrame):
    def __init__(self, master):
        self.master = master
        
        # Title Label
        self.title_label = ctk.CTkLabel(self.master, text="Autoencoder Neural Network", 
                                        anchor="w", font=("bold", 22))
        self.title_label.place(relx=0.015, rely=0.05, relwidth=0.25, relheight=0.05)
        
        # Entry Fields
        self.learning_rate_label = ctk.CTkLabel(self.master, text="Learning Rate:", font=("normal", 12))
        self.learning_rate_label.place(relx=0.015, rely=0.15, relwidth=0.2, relheight=0.05)
        self.learning_rate_entry = ctk.CTkEntry(self.master, font=("normal", 12))
        self.learning_rate_entry.place(relx=0.25, rely=0.15, relwidth=0.15, relheight=0.05)
        
        self.momentum_rate_label = ctk.CTkLabel(self.master, text="Momentum Rate:", font=("normal", 12))
        self.momentum_rate_label.place(relx=0.015, rely=0.25, relwidth=0.2, relheight=0.05)
        self.momentum_rate_entry = ctk.CTkEntry(self.master, font=("normal", 12))
        self.momentum_rate_entry.place(relx=0.25, rely=0.25, relwidth=0.15, relheight=0.05)
        
        self.epoch_max_label = ctk.CTkLabel(self.master, text="Max Epochs:", font=("normal", 12))
        self.epoch_max_label.place(relx=0.015, rely=0.35, relwidth=0.2, relheight=0.05)
        self.epoch_max_entry = ctk.CTkEntry(self.master, font=("normal", 12))
        self.epoch_max_entry.place(relx=0.25, rely=0.35, relwidth=0.15, relheight=0.05)
        
        # Buttons
        self.train_button = ctk.CTkButton(self.master, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                           text="Train Network", font=("normal", 14), command=self.train_network, width=180)
        self.train_button.place(relx=0.015, rely=0.45, relwidth=0.15, relheight=0.05)
    
        # Create canvas to display images
        self.image_1 = Image.new("RGB", (320, 320), "white")
    
        self.image_label_1 = ctk.CTkLabel(self.master, text="Plot")
        self.image_label_1.place(relx=0.615, rely=0.1)
        self.image_widget_1 = ctk.CTkLabel(self.master, image=ImageTk.PhotoImage(self.image_1), text="")
        self.image_widget_1.place(relx=0.615, rely=0.15)
        
    
    def train_network(self):
        x_train, _ = get_training_data(r'input\chest_xray\train')
        
        learning_rate = float(self.learning_rate_entry.get())
        momentum_rate = float(self.momentum_rate_entry.get())
        epoch_max = int(self.epoch_max_entry.get())
        
        x_train = x_train[:100]
        train_data = x_train
        
        autoencoder = AutoEncoder()
        loss, latent_space, decoded_inputs = autoencoder.train(train_data, learning_rate, momentum_rate, epoch_max)
        
        # Plot the loss
        plot_loss_ac(loss)
        
        image_1 = Image.open("images\_autoenc\image_1.png").resize((320, 320))
        image_1_tk = ImageTk.PhotoImage(image_1)
        
        self.image_widget_1.configure(image=image_1_tk)
        
        # Plot the images
        # original_image = ...  # Your original image
        # plot_images_ac(original_image, decoded_inputs)