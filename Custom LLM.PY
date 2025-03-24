import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset Preparation
class TextSummarizationDataset(Dataset):
    def __init__(self, filepath, max_length=512):
        self.data = pd.read_csv(filepath)
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['dialogue']
        target_text = self.data.iloc[idx]['summary']
        
        input_ids = self.text_to_ids(input_text)
        target_ids = self.text_to_ids(target_text)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long)
        }

    def text_to_ids(self, text):
        tokens = [ord(char) for char in text[:self.max_length]]  # Convert each character to its ASCII value
        padding = [0] * (self.max_length - len(tokens))
        return tokens + padding

# Paths
train_path = "/home/f200173/NLP_A3/samsum-train.csv"
val_path = "/home/f200173/NLP_A3/samsum-validation.csv"

# Datasets and DataLoaders
max_length = 128
batch_size = 16

train_dataset = TextSummarizationDataset(train_path, max_length=max_length)
val_dataset = TextSummarizationDataset(val_path, max_length=max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Transformer Model Components
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_length):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self._create_positional_encoding(embed_size, max_length)

    def _create_positional_encoding(self, embed_size, max_length):
        pos = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))
        pe = torch.zeros(max_length, embed_size)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1), :].to(x.device)

class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_encoder_layers, num_decoder_layers, max_length):
        super(TransformerSummarizer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_length)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)

        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.positional_encoding(src)
        tgt_emb = self.embedding(tgt) + self.positional_encoding(tgt)

        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_key_padding_mask=tgt_mask)

        return self.fc_out(output)

# Hyperparameters
vocab_size = 256  # Assuming ASCII characters
embed_size = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6

model = TransformerSummarizer(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    max_length=max_length
).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Training Loop
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['target_ids'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            optimizer.zero_grad()
            output = model(src, tgt_input)

            loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(train_loader)}")

        # Validation
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            src = batch['input_ids'].to(device)
            tgt = batch['target_ids'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)
            loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader)}")

# Train the Model
train_model(model, train_loader, val_loader, num_epochs=10)
from rouge_score import rouge_scorer

# Evaluation Function
def evaluate_model(model, dataloader):
    model.eval()
    references, predictions = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt = batch['target_ids'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Generate outputs
            output = model(src, tgt_input)

            # Calculate loss
            loss = criterion(output.view(-1, vocab_size), tgt_output.view(-1))
            total_loss += loss.item()

            # Decode predictions
            preds = output.argmax(dim=-1)
            for ref, pred in zip(tgt, preds):
                references.append(decode_ids_to_text(ref))
                predictions.append(decode_ids_to_text(pred))

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}")

    # Compute ROUGE scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, pred) for ref, pred in zip(references, predictions)]

    avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
    avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

    print(f"ROUGE-1: {avg_rouge1}, ROUGE-L: {avg_rougeL}")
    return references, predictions

# Helper function to decode IDs to text
def decode_ids_to_text(ids):
    text = ''.join([chr(i) for i in ids if i != 0])  # Convert ASCII values back to characters
    return text.strip()

# Evaluate the model on validation data
references, predictions = evaluate_model(model, val_loader)
# Generate Summaries for Test Data
def generate_summary(model, dataloader):
    model.eval()
    summaries = []

    with torch.no_grad():
        for batch in dataloader:
            src = batch['input_ids'].to(device)
            tgt = batch['target_ids'].to(device)

            # Start with <START> token (assuming the first token in src is the start token)
            decoded_sequence = torch.zeros_like(tgt).to(device)
            decoded_sequence[:, 0] = src[:, 0]  # Start decoding with the first token
            
            for t in range(1, decoded_sequence.size(1)):
                output = model(src, decoded_sequence[:, :t])
                next_token = output.argmax(dim=-1)[:, -1]
                decoded_sequence[:, t] = next_token

            # Decode summaries
            for seq in decoded_sequence:
                summaries.append(decode_ids_to_text(seq))

    return summaries

# Generate test summaries
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_summaries = generate_summary(model, test_loader)

# Print Sample Summaries
for i in range(5):  # Display 5 examples
    print(f"Original Dialogue: {decode_ids_to_text(test_dataset[i]['input_ids'])}")
    print(f"Generated Summary: {test_summaries[i]}")
    print(f"Reference Summary: {decode_ids_to_text(test_dataset[i]['target_ids'])}")
    print('-' * 50)
