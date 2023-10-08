# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 22:31:43 2023

@author: Irfan
"""
import numpy as np

import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


root = Tk()

frame = tk.Frame(root)
lbl_heading = tk.Label(frame, text='Animal Recognition', padx=25, pady=25, font=('verdana',16))
lbl_pic_path = tk.Label(frame, text='Image Path:', padx=25, pady=25, font=('verdana',16))
lbl_show_pic = tk.Label(frame)
entry_pic_path = tk.Entry(frame, font=('verdana',16))
btn_browse = tk.Button(frame, text='Selected Image', bg='grey', fg='#ffffff', font=('verdana',16))
lbl_prediction = tk.Label(frame, text='Animal:  ', padx=25, pady=25, font=('verdana',16))
lbl_predict = tk.Label(frame, font=('verdana',16))
lbl_confid = tk.Label(frame, text='Confidence:  ', padx=25, pady=25, font=('verdana',16))
lbl_confidence = tk.Label(frame, font=('verdana',16))



def selectimg():
    global img
    global filename
    filename = filedialog.askopenfilename(initialdir="/my_images", title="Select Image", filetypes=(("jpg images","*.jpg"),("png images","*.png"),("jpeg images","*.jpeg")))
    
    img = Image.open(filename)
    img = img.resize((250,250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    lbl_show_pic['image'] = img
    entry_pic_path.insert(0, filename)
    
    prediction,confidence = predict_animal(filename)
    lbl_predict.config(text=prediction)
    lbl_confidence.config(text=confidence)
    
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img

model = load_model('animal_recognition_model.h5')


class_labels = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'goat', 'horse', 'spider', 'squirrel']

def predict_animal(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    animal = class_labels[predicted_class]
    confidence = prediction[0][predicted_class] * 100

    return animal, confidence


btn_browse['command'] = selectimg
frame.pack()

lbl_heading.grid(row=0, column=0, columnspan="2", padx=10, pady=10)
lbl_pic_path.grid(row=1, column=0)
entry_pic_path.grid(row=1, column=1, padx=(0, 20))
lbl_show_pic.grid(row=2, column=0, columnspan="2")
btn_browse.grid(row=3, column=0, columnspan="2", padx=10, pady=10)
lbl_prediction.grid(row=4, column=0)
lbl_predict.grid(row=4, column=1, padx=2, sticky='w')
lbl_confid.grid(row=5, column=0)
lbl_confidence.grid(row=5, column=1, padx=2, sticky='w')

root.mainloop()








