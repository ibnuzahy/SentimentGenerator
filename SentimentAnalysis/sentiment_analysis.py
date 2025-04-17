import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# Load model dan tokenizer sekali saja
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

def sentiment_analyzer(text_to_analyse):
    """
    Menganalisis sentimen dari string atau list of strings dan mengembalikan dictionary
    dengan format: {kalimat: {'label': ..., 'score': ...}}
    """
    # Kalau input bukan list, ubah jadi list
    if isinstance(text_to_analyse, str):
        text_to_analyse = [text_to_analyse]

    hasil = {}
    for kalimat in text_to_analyse:
        inputs = tokenizer(kalimat, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=1)
        predicted_class_id = logits.argmax().item()
        label = model.config.id2label[predicted_class_id]
        score = probs[0][predicted_class_id].item()
        hasil[kalimat] = {"label": label, "score": round(score, 4)}
    return hasil
