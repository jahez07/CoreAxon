from flask import Flask, render_template, request, redirect
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

model = tf.keras.models.load_model('/Users/jahezabrahamjohny/Documents/GitHub/CoreAxon/autoencoder_model.h5')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    file= request.files['temperature_data']
    df = pd.read_csv(file)
    train_size = int(len(df) * 0.8)  # 80% train, 20% test
    #train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    temperatures = df['temperature'].tolist()
    anomaly_scores = model(temperatures)
    
    # Plotting anomalies
    plt.figure(figsize=(10, 6))

    plt.plot(test_df['timestamp'], test_df['temperature'], marker='o', linestyle='-', color='b', label='Temperature')

# Highlight anomalies (threshold example)
    threshold = np.mean(anomaly_scores) + 2 * np.std(anomaly_scores)
    anomalies = test_df[anomaly_scores > threshold]
    plt.scatter(anomalies['timestamp'], anomalies['temperature'], color='r', marker='x', label='Anomalies')

    plt.xlabel('Timestamp')
    plt.ylabel('Temperature')
    plt.title('Temperature Anomaly Detection with Autoencoder')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
            
    # Convert plot to base64 encoding and embed in HTML
    plot_img = BytesIO()
    plt.savefig(plot_img, format='png')
    plot_img.seek(0)
    plot_url = base64.b64encode(plot_img.getvalue()).decode('utf8')

            # Render the template with prediction results and plot
    return render_template('index.html', plot_url=plot_url)

if __name__ == "__main__":
    app.run(port=5000, debug=True)