from flask import Flask, render_template, request, redirect
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64

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
    anomaly_scores = model(temperatures)

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(temperatures, anomaly_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Temperature')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Detection')
    plt.grid(True)
            
    # Convert plot to base64 encoding and embed in HTML
    plot_img = BytesIO()
    plt.savefig(plot_img, format='png')
    plot_img.seek(0)
    plot_url = base64.b64encode(plot_img.getvalue()).decode('utf8')

            # Render the template with prediction results and plot
    return render_template('index.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(port=3000, debug=True)