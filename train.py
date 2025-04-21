import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaModel,
    RobertaPreTrainedModel,
    get_linear_schedule_with_warmup,
    AutoConfig,
)
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as HFDataset
import nlpaug.augmenter.word as naw
import time
import logging
from sklearn.metrics import accuracy_score

# === Device and Logging ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# === Hyperparameters ===
MAX_LEN = 256
BATCH_SIZE = 256
EPOCHS = 3
LEARNING_RATE = 1e-5
LORA_R = 2
LORA_ALPHA = 4
LORA_DROPOUT = 0.05
EARLY_STOPPING_PATIENCE = 3
USE_FNN = False
USE_AUGMENTATION = True

# === Load dataset ===
dataset = load_dataset("ag_news")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    dataset["train"]["text"], dataset["train"]["label"], test_size=0.1, random_state=42
)

# === Tokenizer ===
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# === Augmentation ===
def augment_text(example):
    try:
        aug = naw.SynonymAug(aug_src='wordnet')
        augmented = aug.augment(example["text"], n=1)
        return {"text": augmented[0] if isinstance(augmented, list) else augmented}
    except Exception:
        return {"text": example["text"]}

if USE_AUGMENTATION:
    raw_dataset = load_dataset("ag_news")
    subset = raw_dataset["train"]
    augmented = subset.map(augment_text)
    combined_data = subset.to_list() + augmented.to_list()
    combined_dataset = HFDataset.from_dict({
        'text': [item['text'] for item in combined_data],
        'label': [item['label'] for item in combined_data]
    })
    train_dataset = combined_dataset
    train_texts = train_dataset['text']
    train_labels = train_dataset['label']

# === Custom Dataset ===
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

# === Model Options ===
class RobertaWithFNN(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_labels),
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        return type('Output', (), {'loss': loss, 'logits': logits})

# === Model Setup ===
if USE_FNN:
    model_config = AutoConfig.from_pretrained("roberta-base", num_labels=4)
    model = RobertaWithFNN(model_config).from_pretrained("roberta-base", config=model_config)
else:
    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)

# === Apply LoRA ===
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["query", "value"],
    bias="none",
)
model = get_peft_model(model, lora_config)

# === Freeze Base Model ===
for name, param in model.named_parameters():
    if "lora" not in name and "classifier" not in name:
        param.requires_grad = False

model.to(device)

# === Dataloaders ===
train_loader = DataLoader(NewsDataset(train_texts, train_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

# === Optimizer & Scheduler ===
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# === Training ===
best_val_acc = 0.0
early_stopping_counter = 0
best_model_path = "best_lora_finetuned_roberta_final"

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()

    logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Log batch progress every 10 batches
        if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
            logger.info(
                f"[Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    avg_train_loss = total_loss / len(train_loader)
    elapsed = time.time() - start_time
    logger.info(f"End of Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f} - Time: {elapsed:.2f}s")

    # === Validation ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model.save_pretrained(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        early_stopping_counter = 0
        logger.info(f"Saved best model at epoch {epoch+1}")
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            logger.info("Early stopping triggered")
            break

# === MC Dropout Inference ===
def mc_dropout_predict(model, dataset, collate_fn, device, iterations=10):
    model.train()
    loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)
    all_logits = []

    for _ in range(iterations):
        logits_per_iter = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_iter.append(outputs.logits.cpu().numpy())
        all_logits.append(np.concatenate(logits_per_iter, axis=0))

    mean_logits = np.mean(np.array(all_logits), axis=0)
    return np.argmax(mean_logits, axis=1)

logger.info("Evaluating best model with MC Dropout")

model = RobertaForSequenceClassification.from_pretrained(best_model_path).to(device)

val_dataset_tokenized = NewsDataset(val_texts, val_labels, tokenizer, MAX_LEN)

def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["label"] for b in batch]),
    }

mc_preds = mc_dropout_predict(model, val_dataset_tokenized, collate_fn, device)
mc_val_acc = accuracy_score(val_labels, mc_preds)
logger.info(f"MC Dropout Validation Accuracy: {mc_val_acc:.4f}")
