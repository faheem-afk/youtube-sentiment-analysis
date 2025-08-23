import pandas as pd
from datetime import datetime
import torch
from utils import get_encoding
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
import io, base64
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import json
import mlflow


app = Flask(__name__)
CORS(app)


labels= {0:'negative', 1: "neutral", 2: "positive"}
file = open("experiment_info.json", 'r')
model_uri = json.load(file)['artifact_uri']
model = mlflow.pytorch.load_model(model_uri)
model.to('cpu')
model.eval()
stopwords = set(STOPWORDS)


@app.route('/', methods=['GET'])
def home():
    return {"message": "hello"}


@torch.inference_mode()
def encoding_text(text):
    inputs = get_encoding(text)
    inputs = {k: v.to('cpu') for k, v in inputs.items()}
    return inputs


@torch.inference_mode()
def predict_labels(text):
    n = len(text)
    final_predictions = []
    for i in range(0, n, 64):
        
        inputs = encoding_text(text[i: i+64])
        
        outputs = model(**inputs)

        predicted_classes_ = outputs.logits.argmax(dim=1).tolist()

        final_predictions.extend(predicted_classes_)

    return final_predictions


def word_c(text):
    
    text = ' '.join(text)
    # Create a word cloud object
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='black',
        colormap='Blues',
        collocations=False , 
        stopwords=stopwords
    ).generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide axes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


def pie_image_base64(counts):
    # Keep label order consistent
    labels = ["positive", "neutral", "negative"]
    sizes  = [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)]
    colors = ["#35c759", "#9097a6", "#ff453a"]  # green, gray, red

    fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=220)
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        textprops={"color": "white", "fontsize": 9}
    )
    ax.axis("equal")
    fig.patch.set_alpha(0)  # transparent bg
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
    

def time_series_plot(final_dict):
    
    df = pd.DataFrame(final_dict).T
    df.index=pd.to_datetime(pd.DataFrame(final_dict).T.index)
    # Plot
    plt.style.use("seaborn-v0_8-whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["positive"], marker="o", label="Positive", color="green")
    plt.plot(df.index, df["neutral"], marker="o", label="Neutral", color="gray")
    plt.plot(df.index, df["negative"], marker="o", label="Negative", color="red")

    plt.title("Monthly Sentiment Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Comments")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=False)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.json
    
    texts = [comment['text'] for comment in data['comments']]
    timestamps = [comment['timestamp'] for comment in data['comments']]
    final_predictions = predict_labels(texts)
    
    month_days = [datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m") \
        for timestamp in timestamps]

    sentiments = list(map(lambda x: labels[x], final_predictions))
    zipped = zip(sentiments, month_days)
    
    sorted_list = sorted(zipped, key=lambda x: x[1])
    
    final_dict = {}
    time_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
    for sent, time in sorted_list:
        if len(final_dict) == 0:
            final_dict[time] = {}
            
        if final_dict.get(time, 0) == 0:
            final_dict[list(final_dict.keys())[-1]] = time_dict
            final_dict[time] = {}
            time_dict = {'positive': 0, 'neutral': 0, 'negative': 0}
            time_dict[sent] += 1
        else:
            time_dict[sent] += 1
    final_dict[time] = time_dict
        
    counts = Counter(sentiments) 
    pie_b64 = pie_image_base64(counts)
    word_cloud_b64 = word_c(texts)
    time_series_b64 = time_series_plot(final_dict)
        
    return jsonify({
                    "pie_url": f"data:image/png;base64,{pie_b64}",
                    "counts": counts,
                    "sentiments": sentiments,
                    "word_cloud_url": f"data:image/png;base64,{word_cloud_b64}",
                    "time_series_url": f"data:image/png;base64,{time_series_b64}",
                    "texts": texts
                    })
        

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=5000)







