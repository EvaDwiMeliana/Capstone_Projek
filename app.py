#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = load_model("C:/Users/Rafidatus Salsabilah/OneDrive/Documents/Skin Detection/model_baru.h5")
print(model)

print("Model Loaded Successfully")

def pred_skin(skin_plant):
  test_image = load_img(skin_plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "Cacar_Air", 'Cacar_Air.html'
       
  elif pred==1:
      return "Cacar_Monyet", 'Cacar_Monyet.html'
        
  elif pred==2:
      return "Campak", 'Campak.html'
        
  elif pred==3:
      return "Normal", 'Normal.html'

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')

@app.route("/deteksi", methods=['GET', 'POST'])
def det():
        return render_template('deteksi.html')

@app.route("/chatbot", methods=['GET', 'POST'])
def cht():
    return render_template("Chatbot.html")
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fet input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_skin(skin_plant=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=5000) 
    