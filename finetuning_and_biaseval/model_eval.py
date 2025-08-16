'''Model evaluation with cured sentences'''
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel, PeftConfig


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ADAPTER_PATH = "/content/fine_tuned_model"

# Adapter configuration
peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForSequenceClassification.from_pretrained(
    peft_config.base_model_name_or_path,
    num_labels=2
)
# fine-tuned model and tookenizer
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

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
    print("Logits:", outputs.logits.detach().cpu().numpy())
    pred = torch.argmax(outputs.logits, dim=1).item()
    if pred == 1:
      return "SUBJ"
    elif pred == 0:
      return "OBJ"
    else:
      return 'NONE'

for sent, expected in test_sentences:
    pred = classify_text(model, tokenizer, sent)
    print(f"Text: {sent}\nPred: {pred}, Expected: {expected}\n")
