import pickle # Untuk menyimpan dan membaca data ke dalam atau dari suatu file berformat .pkl
from flask import Flask, render_template, request, jsonify
import pymysql
from keras.preprocessing.image import load_img, img_to_array
import numpy as np # Memudahkan analisis dan komputasi matematika tanpa perlu menulis kode dari awa
import os
import pandas as pd
import string
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
# Mengubah kata berimbuhan bahasa Indonesia menjadi bentuk dasarnya.
import nltk # Untuk pengolahan data bahasa manusia.
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.models import load_model
import json
# Menghasilkan representasi JavaScript Object Notation (JSON) struktur data sebagai teks sehingga cocok untuk menyimpan atau mentransmisikan di seluruh jaringan.
import random
from nltk.stem import WordNetLemmatizer

# Load the NLTK popular resources
nltk.download('popular')

# Create Flask instance
app = Flask(__name__)

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'salsa',
    'database': 'analisis2',
}

# Function to insert data into MySQL
def insert_data_to_mysql(data):
    connection = pymysql.connect(**db_config)
    try:
        with connection.cursor() as cursor:
            # Modify the SQL statement to include id_review as auto-increment
            sql = """
                CREATE TABLE IF NOT EXISTS input_review (
                # Membuat tabel 'input_review' jika belum ada.
                    id_review INT AUTO_INCREMENT PRIMARY KEY,
                    nama VARCHAR(255) NOT NULL,
                    tanggal DATE NOT NULL,
                    review TEXT NOT NULL
                )
            """
            cursor.execute(sql)

            # Insert data into the table
            sql_insert = "INSERT INTO input_review (nama, tanggal, review) VALUES (%s, %s, %s)"
            # Menyisipkan baris baru ke dalam tabel 'input_review' dengan nilai dari kamus data.
            cursor.execute(sql_insert, (data['nama'], data['tanggal'], data['review']))
        connection.commit()
        # Untuk melakukan perubahan pada database.
        print("Data inserted successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        connection.close()

# Enable CORS for all routes
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

app.after_request(add_cors_headers)


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
# Membersihkan kalimat dengan melakukan tokenisasi dan lemmatisasi
    sentence_words = nltk.word_tokenize(sentence)
    # Tokenisasi kalimat menjadi kata-kata menggunakan fungsi word_tokenize
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # Mengonversi setiap kata menjadi huruf kecil terlebih dahulu
    return sentence_words
    # Mengembalikan daftar kata-kata yang telah di-tokenisasi dan dilemmatisasi.

def bow(sentence, words, show_details=True):
# Menerima sebuah kalimat (sentence), daftar kata-kata yang ingin dijadikan fitur BoW (words), dan parameter show_details yang mendefault ke True.
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    # Membuat array dengan panjang sejumlah kata yang diinginkan
    for s in sentence_words:
    # Melakukan iterasi melalui setiap kata dalam (sentence_words) dari kalimat yang telah dibersihkan.
        for i, w in enumerate(words):
        # Melakukan iterasi melalui kata (words) bersama dengan indeksnya (i) menggunakan fungsi enumerate
            if w == s:
            #  Memeriksa apakah kata dalam kalimat (s) sama dengan kata dalam daftar kata-kata yang diinginkan (w).
                bag[i] = 1
                # Jika kondisi tersebut benar,array bag diatur menjadi 1, menandakan bahwa kata tersebut ada dalam kalimat.
                if show_details:
                # Mencetak pesan yang menyatakan bahwa kata tersebut ditemukan dalam bag
                    print("found in bag: %s" % w)
    return np.array(bag) 
    # Mengembalikan representasi BoW dalam bentuk array NumPy

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    # Hasil dari perwakilan bag-of-words dari kalimat menggunakan fungsi bow
    res = model.predict(np.array([p]))[0]
    # Membuat prediksi menggunakan model
    ERROR_THRESHOLD = 0.25
    # Menetapkan ambang batas kesalahan untuk mempertimbangkan prediksi
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Memfilter hasil prediksi berdasarkan ambang batas kesalahan
    results.sort(key=lambda x: x[1], reverse=True)
    # Mengurutkan hasil prediksi berdasarkan peluang secara menurun
    return_list = []
    for r in results:
         return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        # Membuat daftar hasil prediksi dalam bentuk kamus
    return return_list
    # Mengembalikan daftar hasil prediksi

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    # Mengambil tag intent dari hasil prediksi
    list_of_intents = intents_json['intents']
    # Mengakses daftar intents dari data JSON
    for i in list_of_intents:
        if i['tag'] == tag:
    #   # Iterasi melalui daftar intents untuk mencari yang sesuai dengan tag
            result = random.choice(i['responses'])
            # Memilih secara acak satu dari respons yang terkait dengan intent tersebut
            break
    return result
    # Mengembalikan respons yang dipilih


def chatbot_response(msg):
    ints = predict_class(msg, chatbot_model)
    # Memprediksi intent dari pesan menggunakan fungsi predict_class dan model chatbot
    res = get_response(ints, intents)
    # Mendapatkan respons dari fungsi get_response berdasarkan hasil prediksi dan daftar intents
    return res
    # Mengembalikan respons yang diperoleh

# Function to predict skin 
def pred_skin(skin_plant):
    test_image = load_img(skin_plant, target_size=(150, 150))
    # Memuat gambar dan menetapkan ukuran target
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255 # pelatihan model deep learning
    # Nilai piksel dalam citra umumnya berkisar dari 0 hingga 255, di mana 0 adalah warna hitam dan 255 adalah warna putih
    # Mengonversi gambar menjadi array dan normalisasi nilai piksel ke rentang [0, 1]
    test_image = np.expand_dims(test_image, axis=0)
    # Mengubah dimensi array dari 3D menjadi 4D

    # Melakukan prediksi menggunakan model
    result = model.predict(test_image)
    
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    # Mengambil indeks kelas dengan nilai probabilitas tertinggi
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
    
@app.route('/input')
def input():
    return render_template('analisis2.html')

# Route to handle form submission
@app.route('/input', methods=['POST', 'OPTIONS'])
def submit_form():
    if request.method == 'OPTIONS':
        # Preflight request, respond successfully
        return jsonify({'status': 'success'})

    data_to_insert = request.get_json()
    insert_data_to_mysql(data_to_insert)
    return jsonify({'status': 'success'})

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
    # Memeriksa apakah permintaan yang diterima adalah metode POST
        file = request.files['image']
        # Mengambil file gambar dari permintaan
        filename = file.filename
        print("@@ Input posted = ", filename)

        # Menyimpan file gambar di lokasi tertentu (static/upload)
        file_path = os.path.join('static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        # Melakukan prediksi kelas menggunakan fungsi pred_skin
        pred, output_page = pred_skin(skin_plant=file_path)
        # Merender halaman output dengan menampilkan hasil prediksi dan gambar pengguna
        return render_template(output_page, pred_output=pred, user_image=file_path)

@app.route("/analisis", methods=["GET", "POST"])
def analisis():
    if request.method == "POST":
        content = request.form.get("input-data")
        # Mendapatkan data dari formulir HTML dengan nama "input-data"
        df_content = pd.DataFrame([content], columns=["content"])
        # Membuat DataFrame menggunakan pandas dengan satu kolom bernama "content"

        # Case Folding
        def clean_content(content):
            content = content.lower()  # convert to lowercase
            content = re.sub("[^a-z]", " ", content)  # hapus semua karakter kecuali a-z
            content = re.sub("\t", " ", content)  # ganti tab dengan spasi
            content = re.sub("\n", " ", content)  # ganti baris baru dengan spasi
            content = re.sub("\s+", " ", content)  # ganti lebih dari 1 spasi dengan 1 spasi
            content = content.strip()  # hapus spasi awal dan akhir
            return content

        # Stopword Removal
        def clean_stopword(content):
            # Stopword Sastrawi
            factory = StopWordRemoverFactory()
            stopword_sastrawi = factory.get_stop_words()
            # Menggunakan StopwordRemover dari Sastrawi
            
            content = content.split() # Memisahkan teks menjadi kata-kata
            content = [w for w in content if w not in stopword_sastrawi]  # Menghapus stopword dari teks
            content = " ".join(w for w in content)
            # Menggabungkan semua kata yang bukan stopword kembali menjadi teks

            # Stopword NLTK
            stopword_nltk = set(stopwords.words("indonesian"))
            # Mendapatkan daftar stopword untuk bahasa Indonesia dari NLTK.
            # Mengonversi daftar stopword menjadi set, sehingga pencarian dalam set dapat dilakukan dengan lebih efisien.
            stopword_nltk = stopword_nltk
            content = content.split()
            # Memisahkan teks (variabel content) menjadi kata-kata. Ini mengubah teks menjadi sebuah list yang berisi kata-kata.
            content = [w for w in content if w not in stopword_nltk]
            #  Menggunakan list comprehension untuk memfilter kata-kata yang bukan stopword. Hanya kata-kata yang tidak termasuk dalam daftar stopword yang akan tetap.
            content = " ".join(w for w in content)
            # Menggabungkan kembali kata-kata yang tidak termasuk dalam stopword menjadi satu teks. Ini mengubah list kata-kata menjadi teks dengan memisahkan kata-kata menggunakan spasi.
            return content
            # Mengembalikan teks yang telah dibersihkan dari stopword menggunakan stopword dari NLTK.

        # Stemming
        def clean_stem(content):
            # Stemming Sastrawi
            factory = StemmerFactory()
            # Menggunakan Stemmer dari Sastrawi
            stemmer_sastrawi = factory.create_stemmer()
            # Membuat stemmer menggunakan objek dari StemmerFactory.
            content = stemmer_sastrawi.stem(content)
            # Melakukan stemming pada teks (mengubah kata2 menjadi bentuk dasarnya)
            return content
            # Mengembalikan hasil stemming

        # Membersihkan teks dari stopwords
        stopwords_trove = ["troveskin", "aplikasi"]

        def clean_trove(text):
            temp = text.split() # Memisahkan teks menjadi kata-kata
            temp = [w for w in temp if not w in stopwords_trove]  # Menghapus stopwords dari teks
            temp = " ".join(word for word in temp) # Menggabungkan kembali kata-kata menjadi teks
            return temp # Mengembalikan teks yang telah dibersihkan

        # Fungsi gabungan untuk semua proses
        def sentiment_prediction(df_content):
            # 1. clean content
            df_content = df_content.apply(clean_content)

            # 3. remove stopwords
            df_content = df_content.apply(clean_stopword)

            # 4. stemming
            df_content = df_content.apply(clean_stem)

            # 5. remove trove stopwords
            df_content = df_content.apply(clean_trove)

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