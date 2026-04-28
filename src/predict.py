import joblib
from sentence_transformers import SentenceTransformer

bert_svm = joblib.load("../outputs/models/bert_balanced_svm.pkl")

with open("../outputs/models/bert_model_name.txt") as f:
    model_name = f.read().strip()

bert_model = SentenceTransformer(model_name)

def predict_sms(text):
    embedding = bert_model.encode([text])
    pred = bert_svm.predict(embedding)[0]
    prob = bert_svm.predict_proba(embedding)[0]

    label = "Spam" if pred == 1 else "Ham"
    confidence = prob[pred]

    return label, confidence