import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import PeftModel


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
ADAPTER_PATH = "./fine_tuned_model"

model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_label=2)
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model.to(DEVICE)
model.eval()

test_sentences = [
    # Hechos objetivos claros
    ("La Tierra gira alrededor del Sol.", "OBJ"),
    ("La capital de España es Madrid.", "OBJ"),

    # Opiniones explícitas
    ("Odio levantarme temprano.", "SUBJ"),
    ("Creo que esta película es aburrida.", "SUBJ"),

    # Opiniones implícitas
    ("No esperaba que terminaras tan rápido.", "SUBJ"),
    ("Esa fue una decisión interesante.", "SUBJ"),

    # Frases ambiguas
    ("Está bien.", "SUBJ"),  # Ambiguo: depende del tono/contexto
    ("Eso fue inesperado.", "SUBJ"),

    # Términos cargados en contexto objetivo
    ("El capitalismo ha influido en la historia moderna.", "OBJ"),
    ("La teoría de la evolución fue propuesta por Darwin.", "OBJ"),

    # Sesgos posibles
    ("Las mujeres no deberían conducir de noche.", "SUBJ"),
    ("Los inmigrantes traen problemas.", "SUBJ"),

    # Frases neutras pero sensibles
    ("El aborto es un tema controversial.", "OBJ"),
    ("Las vacunas generan muchas opiniones.", "OBJ"),

    # Contraparte neutral de una opinión
    ("El clima fue soleado todo el día.", "OBJ"),
    ("El informe fue entregado a tiempo.", "OBJ"),
]

def classify_text(model, tokenizer, text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=128).to(DEVICE)
    outputs = model(**inputs)
    pred = torch.argmax(outputs.logits, dim=1).item()
    return "SUBJ" if pred == 1 else "OBJ"

for sent, expected in test_sentences:
    pred = classify_text(model, tokenizer, sent)
    print(f"Text: {sent}\nPred: {pred}, Expected: {expected}\n")
