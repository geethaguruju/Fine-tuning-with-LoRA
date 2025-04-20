import torch
import pandas as pd
import pickle
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model with LoRA
model_path = "best_lora_finetuned_roberta_final"

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the model
model = RobertaForSequenceClassification.from_pretrained(
    model_path, num_labels=4  # Ensure the model is set to 4 classes for AG News
)
model.to(device)

# Load the test data (assumed to be loaded from pickle file)
test_data_path = 'test_unlabelled.pkl'
with open(test_data_path, 'rb') as f:
    test_data = pickle.load(f)

test_texts = test_data['text']  # Assuming the key is 'text', adjust if necessary

# Create DataLoader for test dataset
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }

# Initialize DataLoader for test data
MAX_LEN = 128  # Adjust the maximum length as needed
BATCH_SIZE = 8  # Adjust batch size based on available memory
test_dataset = TestDataset(test_texts, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Inference on the test data
model.eval()
predictions = []
ids = []  # Assuming test data has an 'ID' field

with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference Progress"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        # Save predicted labels and IDs
        predictions.extend(preds.cpu().numpy())
        ids.extend(range(idx * BATCH_SIZE, (idx + 1) * BATCH_SIZE))

# Save predictions to CSV
output_df = pd.DataFrame({
    "ID": ids,
    "Label": predictions
})

# Save the CSV to a desired location
output_csv_path = "predictions.csv"
output_df.to_csv(output_csv_path, index=False)

print(f"Inference completed. Predictions saved to {output_csv_path}")