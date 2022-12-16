import os
from flask import Flask, request, jsonify, render_template
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import os
import glob
import pandas as pd    # to load dataset
import numpy as np     # for mathematic equation
from nltk.corpus import stopwords   # to get collection of stopwords
from deep_translator import GoogleTranslator
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import ktrain 
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer
import tensorflow as tf
#import easyocr
from pickle import load
from sklearn.metrics import accuracy_score
model3 = load(open('Models/finalized_model.pkl', 'rb'))
# load the scaler
scaler = load(open('Models/scaler.pkl', 'rb'))
from tensorflow.keras.preprocessing import image
import urllib.request
#import os
#import cv2
#import pytesseract
#import easyocr
from deep_translator import GoogleTranslator
from langdetect import detect
from googletrans import Translator
translate = Translator()
predictor = ktrain.load_predictor('Models/Model')
Text_Predictor = ktrain.load_predictor('Models/Textclass')

from webdriver_manager.chrome import ChromeDriverManager
#from webdriver_manager.firefox import GeckoDriverManager

import pandas as pd
import time

service=Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)
#driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
#driver = webdriver.Chrome(executable_path=r"C:\Users\kalaiselvan\Documents\chromedriver.exe")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('app_frontend.html', prediction_text="")

@app.route('/predict', methods=['GET','POST'])
def predict():

    a_description = request.form.get('description')
    driver.get(f"{a_description}")
    time.sleep(3)
    SCROLL_PAUSE_TIME = 4
    text = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        tweet = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
        for i in tweet:
            try:
                print(i.get_attribute("innerText"))
                i = i.get_attribute("innerText")
            except:
                i = "cant get this text"
                print("nothingggggggggggggggggggggggggggggggggg")
            try:
                lan = detect(i)
                if lan == "zh-cn":
                   lan = "zh-CN"
                   #print(lan,"lllllllllllllllllllllllllllllllllllllllllllllll")
            except:
                lan = 'en'
            try:
                message = GoogleTranslator(source=lan, target='en').translate(i)
            except:
                message = i
            if len(message) < 5000:
               predicted = Text_Predictor.predict(message)
               print(predicted,"jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
            else:
               predicted = "others"
               print(predicted,"jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
            text.append(predicted)
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 1300);")
        # Wait to load page
        time.sleep(2)
        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
           break
        last_height = new_height
    business = text.count("business")
    mlm = text.count("mlm")
    Motivation = text.count("Motivation")
    Others = text.count("others")
    final_data = [mlm,business,Motivation,Others]
    X_test = np.array(final_data)
    X_test = scaler.transform(X_test.reshape(1, -1))
    y = model3.predict(X_test)
    return render_template('app_frontend.html', prediction_text=y)



if __name__ == "__main__":
    #app.run()

    #app.debug = True
    app.run(host='0.0.0.0', port=5000)
