from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from datasets import load_dataset

MODEL_PATH = './fine_tuned_model'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
DATASET_NAME = 'dataset'

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

val_dataset = load_dataset('csv', data_files={
    'validation': f'/content/{DATASET_NAME}/dev.tsv'
    }, delimiter="\t" )
val_dataset = val_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.remove_columns(["tweet_id", "text"])
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

preds = []
labels = []

with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch['input_ids'].to(DEVICE)
        att_mask = batch['attention_mask'].to(DEVICE)
        label = batch['label'].cpu().numpy()

        output = model(input_ids=input_ids, attention_mask=att_mask)
        pred = torch.argmax(output.logits, dim=1).cpu().numpy()

        preds.append(pred)
        labels.append(label)

print(classification_report(labels, preds, target_names=['SUBJ', 'OBJ'])) 