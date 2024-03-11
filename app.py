import numpy as np 
import gradio as gr
import tensorflow as tf
import cv2
import keras
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.pyplot as plt
import requests

generator = tf.keras.models.load_model("saved_model/20epochs.h5")
SIZE = 256

def format_user_input(user_input_path):
  img = cv2.imread(user_input_path,1) 
  # # open cv reads images in BGR format so we have to convert it to RGB
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # #resizing image
  img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
  # img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
  img = img.astype('float32') / 255.0
  img = img_to_array(img)
  return img



 


def colorize(input_img):
  # return format_user_input(input_img)
  user_input_arr = []
  user_input_arr.append(format_user_input(input_img))
  user_input_arr.append(format_user_input(user_input_path="user_input/pimbahal.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/kagbeni.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/jaljala.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/patan.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/patandhoka.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/valley.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/buddha.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/chitwan.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2040.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2032.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2024.jpg"))
  user_input_arr.append(format_user_input(user_input_path="user_input/2016.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2048.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2056.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2072.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2080.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2088.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2096.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2104.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2112.jpg"))
  user_input_arr.append(format_user_input(user_input_path="landscape Images/gray/2200.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2200.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2201.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2202.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2203.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2204.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2205.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2206.jpg"))
  user_input_arr.append(format_user_input("landscape Images/gray/2207.jpg"))
  user_input_dataset = tf.data.Dataset.from_tensor_slices(np.array(user_input_arr)).batch(8)

  for example_input in (user_input_dataset).take(1):
    prediction = generator( example_input, training=True)

  return np.array(prediction[0])
  
demo = gr.Interface(colorize, gr.Image(type="filepath"), gr.Image(height=400, width=400))
demo.launch()