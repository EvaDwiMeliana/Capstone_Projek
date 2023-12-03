import pickle
from flask import Flask, render_template, request
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import pandas as pd
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.models import load_model
import json
import random
from nltk.stem import WordNetLemmatizer

# Load the NLTK popular resources
nltk.download('popular')

# Create Flask instance
app = Flask(__name__)

# Load model
model = load_model("model_baru.h5")

# Load other models and data
vectorizer = pickle.load(open("vectorizer_tfidf.pkl", "rb"))
svc = pickle.load(open("model_svc.pkl", "rb"))

chatbot_model = load_model("model_chtbot.h5")
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()



def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, chatbot_model)
    res = get_response(ints, intents)
    return res

# Function to predict skin disease
def pred_skin(skin_plant):
    test_image = load_img(skin_plant, target_size=(150, 150))  # load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimension 3D to 4D


    # Now, use the features as input for chatbot_model
    result = model.predict(test_image)
    
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)  # get the index of the maximum value
    print(pred)
    if pred == 0:
        return "Cacar_Air", 'Cacar_Air.html'  # if index is 0, it's chickenpox
    elif pred == 1:
        return "Cacar_Monyet", 'Cacar_Monyet.html'  # if index is 1, it's monkeypox
    elif pred == 2:
        return "Campak", 'Campak.html'  # if index is 2, it's measles
    elif pred == 3:
        return "Normal", 'Normal.html'  # if index is 3, it's normal

# Flask routes

@app.route("/chatbot", methods=['GET', 'POST'])
def cht():
    return render_template("Chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/deteksi", methods=['GET', 'POST'])
def det():
    return render_template('deteksi.html')



@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # get input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/upload', filename)  # Image to be predicted
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = pred_skin(skin_plant=file_path)
        return render_template(output_page, pred_output=pred, user_image=file_path)

@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    if request.method == "POST":
        content = request.form.get("input-data")
        df_content = pd.DataFrame([content], columns=["content"])

        # Case Folding
        def clean_content(content):
            content = content.lower()  # convert to lowercase
            content = re.sub("[^a-z]", " ", content)  # remove all characters except a-z
            content = re.sub("\t", " ", content)  # replace tab with space
            content = re.sub("\n", " ", content)  # replace new line with space
            content = re.sub("\s+", " ", content)  # replace more than 1 space with 1 space
            content = content.strip()  # remove leading and trailing spaces
            return content

        # Stopword Removal
        def clean_stopword(content):
            # Stopword Sastrawi
            factory = StopWordRemoverFactory()
            stopword_sastrawi = factory.get_stop_words()
            content = content.split()  # split into words
            content = [w for w in content if w not in stopword_sastrawi]  # remove stopwords
            content = " ".join(w for w in content)  # join all words that are not stopwords

            # Stopword NLTK
            stopword_nltk = set(stopwords.words("indonesian"))
            stopword_nltk = stopword_nltk
            content = content.split()  # split into words
            content = [w for w in content if w not in stopword_nltk]  # remove stopwords
            content = " ".join(w for w in content)  # join all words that are not stopwords
            return content

        # Stemming
        def clean_stem(content):
            # Stemming Sastrawi
            factory = StemmerFactory()
            stemmer_sastrawi = factory.create_stemmer()
            content = stemmer_sastrawi.stem(content)
            return content

        # Clean Tokped
        stopwords_tokped = ["tokopedia", "aplikasi"]

        def clean_tokped(text):
            temp = text.split()  # split words
            temp = [w for w in temp if not w in stopwords_tokped]  # remove stopwords
            temp = " ".join(word for word in temp)  # join all words
            return temp

        # Combined function for all processes
        def sentiment_prediction(df_content):
            # 1. clean content
            df_content = df_content.apply(clean_content)

            # 3. remove stopwords
            df_content = df_content.apply(clean_stopword)

            # 4. stemming
            df_content = df_content.apply(clean_stem)

            # 5. remove Tokped stopwords
            df_content = df_content.apply(clean_tokped)

            # 6. vectorizer
            df_content = vectorizer.transform(df_content)

            # 7. predict
            sentiment_value = svc.predict(df_content)

            return sentiment_value

        # Perform prediction using the "sentiment_prediction" function
        prediction_value = sentiment_prediction(df_content["content"])

        if prediction_value[0] == 1:
            return render_template("analisis.html", data="Positif", content=content)
        else:
            return render_template("analisis.html", data="Negatif", content=content)

    return render_template("analisis.html")

# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False, port=5000)
