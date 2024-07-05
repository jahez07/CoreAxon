from flask import Flask, render_template, request, redirect
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model('/Users/jahezabrahamjohny/Documents/GitHub/CoreAxon/autoencoder_model.h5')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file= request.files['temprature data']
    df = pd.read_csv(file)
    temperatures = df['temperature'].tolist()
    prediction = model(temperatures)

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(port=3000, debug=True)