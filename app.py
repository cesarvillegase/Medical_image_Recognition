from typing import Tuple
import numpy as np
import pandas as pd
import customtkinter as ctk
import matplotlib.pyplot as plt
import os
import webbrowser
from PIL import Image

from tabs.preprocessing_tab import PreprocessingTab

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Medical Image Recognition")
        self.geometry(f"{1280}x{640}")
        
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1) 
        
        self.create_sidebar_frame()
        self.create_tabview()
        
    def create_sidebar_frame(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(6, weight=1)
        
        self.setup_sidebar_widgets()
        
    def setup_sidebar_widgets(self):
        logo_label = ctk.CTkLabel(self.sidebar_frame, text="Medical Image \n Recognition",
                                  font=("bold", 18))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Create the label for the author and the description
        author_label = ctk.CTkLabel(self.sidebar_frame, text="Created by: \n Cesar Alejandro \n Villegas Espindola",
                                    font=("normal", 14))
        author_label.grid(row=1, column=0, padx=20, pady=(10))
        
        # Create textbox
        self.textbox = ctk.CTkTextbox(self.sidebar_frame)
        self.textbox.place(relx=0.05, rely=0.25, relwidth=0.9, relheigh=0.45)
        
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        
        with open(dir_path + '\\Medical_image_Recognition\\desc\\description.txt', 'r') as file:
            description_file = file.readlines()
        
        description_file = ''.join(line for line in description_file)
        
        self.texts = {"Description" : description_file}    
        self.textbox.insert('0.0', self.texts["Description"])
        self.textbox.configure(state="disabled", wrap="word")    
    
        # GITHUB ICON
        
        github_image = ctk.CTkImage(light_image=Image.open("images/sidebar_frame/icon_github.png"),
                                  dark_image=Image.open("images/sidebar_frame/icon_github.png"),
                                  size=(30, 30))
        button_github = ctk.CTkButton(self.sidebar_frame, text= 'Github', 
                                                image=github_image, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_github)
        button_github.place(relx=0.025, rely=0.75, relwidth=0.9, relheight=0.05)
        
        # LINKEDIN ICON
        linkedin = ctk.CTkImage(light_image=Image.open("images/sidebar_frame/icon_linkedin.png"),
                                  dark_image=Image.open("images/sidebar_frame/icon_linkedin.png"),
                                  size=(30, 30))
        button_linkedin = ctk.CTkButton(self.sidebar_frame, text= 'LinkedIn', 
                                                image=linkedin, fg_color='transparent', text_color=('black', 'white'),
                                                command=self.open_linkedin)
        button_linkedin.place(relx=0.025, rely=0.825, relwidth=0.9, relheight=0.05)
    
        exit_button = ctk.CTkButton(self.sidebar_frame, fg_color="#f77f00", hover_color="#fcbf49", text_color="#000000", 
                                         text="Close App", font=("bold", 14), command=self.exit,
                                         width=180)
        exit_button.grid(row=8, column=0, padx=20, pady=(20, 10))
    
    def open_github(self):
        webbrowser.open_new_tab('https://github.com/cesarvillegase')
    
    def open_linkedin(self):
        webbrowser.open_new_tab('www.linkedin.com/in/cesarvillegas-esp')
    
    def create_tabview(self):
        # Create tabview
        tabview = ctk.CTkTabview(self, width=250)
        tabview.grid(row=0, column=1, padx=(10, 0), pady=(10, 0), sticky="nsew")
        self.setup_tabs(tabview)
        
    def setup_tabs(self, tabview):
        tabs = ["Preprocessing", "Convolution", "Backpropagation", "AutoEncoder"]
        for tab_name in tabs:
            tab = tabview.add(tab_name)
            if tab_name == "Preprocessing":
                preprocessing_tab = PreprocessingTab(master=tab)
            '''
            if tab_name == "Convolution":
                self.setup_convolution_tab(tab)
            if tab_name == "Backpropgation":
                self.setup_backprop_tab(tab)
            if tab_name == "AutoEncoder":
                self.setup_autoencoder(tab)
            '''
        
    def exit(self):
        self.destroy()    
        
if __name__ == "__main__":
    app = App()
    app.mainloop()