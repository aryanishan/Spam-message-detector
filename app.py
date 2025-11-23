from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

# Load trained model
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocess (SAME as notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    msg = request.json["message"]
    cleaned = clean_text(msg)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)[0]
    return jsonify({"prediction": "SPAM" if pred == 1 else "HAM"})

if __name__ == "__main__":
    app.run(debug=True)
