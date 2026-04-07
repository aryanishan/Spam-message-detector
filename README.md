# 📩 Spam SMS Detector

A machine learning-based web application that classifies SMS messages as **Spam** or **Not Spam (Ham)**. This project helps users identify unwanted or fraudulent messages efficiently using Natural Language Processing (NLP) techniques.

---

## 🚀 Features

* 🔍 Classifies SMS messages as **Spam** or **Ham**
* ⚡ Fast and accurate prediction using trained ML model
* 🧠 Uses NLP techniques for text preprocessing
* 🌐 Simple and interactive user interface
* 📊 Lightweight and efficient model

---

## 🛠️ Tech Stack

* **Frontend:** HTML, CSS, JavaScript *(or React if used)*
* **Backend:** Python (Flask / Django)
* **Machine Learning:** Scikit-learn
* **NLP:** NLTK / TF-IDF Vectorizer
* **Dataset:** SMS Spam Collection Dataset

---

## 📂 Project Structure

```
Spam-SMS-Detector/
│
├── data/                 # Dataset files
├── model/                # Trained ML model
├── static/               # CSS, JS files
├── templates/            # HTML templates
├── app.py                # Main application file
├── vectorizer.pkl        # Saved vectorizer
├── model.pkl             # Saved model
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/spam-sms-detector.git
cd spam-sms-detector
```

2. **Create virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the application**

```bash
python app.py
```

5. Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## 🧠 How It Works

1. Input SMS text from the user
2. Preprocess text (remove stopwords, punctuation, etc.)
3. Convert text into numerical format using **TF-IDF Vectorization**
4. Pass data into trained ML model
5. Output prediction:

   * ✅ Ham (Not Spam)
   * 🚫 Spam

---

## 📊 Model Details

* Algorithm used: **Naive Bayes / Logistic Regression** *(update based on your model)*
* Accuracy: ~ **95%+** *(update if you have exact value)*
* Feature Extraction: **TF-IDF Vectorizer**

---

## 📸 Screenshots *(Optional)*

*Add screenshots of your UI here*

---

## 🔮 Future Improvements

* 📱 Mobile-friendly UI
* 📊 Show probability/confidence score
* 🔄 Real-time message filtering
* ☁️ Deploy on cloud (AWS / Render / Vercel)

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Aryan Ishan**

* Passionate about Data Science & AI
* Building real-world ML projects

---

