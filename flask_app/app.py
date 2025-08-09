import torch
import mlflow
from transformers import AutoTokenizer
from utils import get_encoding
from flask import Flask, jsonify, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return {"message": "hello"}


@app.route('/predict', methods=['POST'])
def predict():
    
    model = mlflow.pytorch.load_model("./sent-bert/model")
    
    data = request.json
    inputs = get_encoding(data['text'])

    outputs = model(**inputs)

    probs = outputs.logits.softmax(dim=1)

    predicted_classes_ = probs.argmax(dim=1).tolist()

    labels= {0:'negative', 1: "neutral", 2: "positive"}

    sentiments = list(map(lambda x: labels[x], predicted_classes_))
    return jsonify({"sentiment": sentiments})
    

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5003)







