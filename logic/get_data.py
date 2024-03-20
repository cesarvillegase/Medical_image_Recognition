import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

def get_training_data(data_dir):
    images = [] 
    labels_list = []
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                images.append(resized_arr)
                labels_list.append(class_num)
            except Exception as e:
                print(e)
    return np.array(images), np.array(labels_list)

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150

x_train, y_train = get_training_data(r'input\chest_xray\train')
x_test, y_test = get_training_data(r'input\chest_xray\test')
x_val, y_val = get_training_data(r'input\chest_xray\val')

def plot_label_distribution(labels, title):
    unique_labels, counts = np.unique(labels, return_counts=True)
    colors = ['#ffb703', '#003f88']  # Specify custom colors for each label
    plt.bar(unique_labels, counts, tick_label=unique_labels, color=colors)
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.show()

# Assuming y_train contains label data
# Plot label distribution for training data
plot_label_distribution(y_train, 'Training Label Distribution')

# Assuming x_train contains image data and y_train contains label data
# Plot the first image
plt.figure(figsize=(5, 5))
plt.imshow(x_train[0], cmap='gray')
plt.title(labels[y_train[0]])  # Get the corresponding label from the labels list
# plt.show()

# Plot the last image
plt.figure(figsize=(5, 5))
plt.imshow(x_train[-1], cmap='gray')
plt.title(labels[y_train[-1]])  # Get the corresponding label from the labels list
# plt.show()

x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_test = np.array(x_test) / 255

print(x_train.shape)

'''
# resize data for deep learning 
x_train = x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val = x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

x_test = x_test.reshape(-1, img_size, img_size, 1)
y_test = np.array(y_test)

print(x_train.shape) 

'''