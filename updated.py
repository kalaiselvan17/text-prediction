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
from pickle import load
from sklearn.metrics import accuracy_score
model3 = load(open('Models/finalized_model.pkl', 'rb'))
# load the scaler
scaler = load(open('Models/scaler.pkl', 'rb'))
from tensorflow.keras.preprocessing import image
import urllib.request
#import cv2
#import pytesseract
#import easyocr
from deep_translator import GoogleTranslator
from langdetect import detect
from googletrans import Translator
translate = Translator()
#predictor = ktrain.load_predictor('Models/Model')
Text_Predictor = ktrain.load_predictor('Models/Textclass')

from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
#from webdriver_manager.firefox import GeckoDriverManager

import pandas as pd
import time

chrome_options = Options()
driver=webdriver.Chrome(ChromeDriverManager().install(),chrome_options=chrome_options)
driver.maximize_window()
#driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
#driver = webdriver.Chrome(executable_path=r"C:\Windows\WinSxS\x86_netfx4-browser_files_b03f5f7f11d50a3a_4.0.15805.0_none_04f1e78822144171\chrome.browser")
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
    URL = []
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        #driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        tweet = driver.find_elements(By.XPATH,"//div[@data-testid='tweetText']")
        for i in tweet:
            try:
                #print(i.get_attribute("innerText"))
                txt = i.get_attribute("innerText")
            except:
                txt = "cant get this text"
                print("nothingggggggggggggggggggggggggggggggggg")
            try:
                lan = detect(txt)
                if lan == "zh-cn":
                   lan = "zh-CN"
                   #print(lan,"lllllllllllllllllllllllllllllllllllllllllllllll")
            except:
                lan = 'en'
            try:
                message = GoogleTranslator(source=lan, target='en').translate(txt)
            except:
                message = txt
            if len(message) < 5000:
               predicted = Text_Predictor.predict(message)
               print(predicted)
            else:
               predicted = "others"
               print(predicted)
            #print(message)
     
            message = message.split()
            print(message)
            g = [x for x in message if re.findall("@", x)]
            for i in g:
                text1 = i.replace("@","")
                j = "https://twitter.com/" + text1
                URL.append(j)
            print(URL)
            #print(g,'kalaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaiiiiiiiiiiiiiiiiiiiiiiii')    
            #text1.extend(g)
            #print(text1)
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
    app.run(host='192.168.1.106', port=5000)





