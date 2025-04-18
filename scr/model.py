""" This file contains the model and functions or classes 
    for its training and inference. """
import torch
from transformers import DistilBertModel, DistilBertConfig
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.nn import functional as F

class DistilBertForSentiment(nn.Module):
    def __init__(self, tokenizer, num_labels):
        """
        Initializes the model with a pre-trained DistilBERT backbone and a classification head.
        
        """
        super(DistilBertForSentiment, self).__init__()
        # Load the pre-trained DistilBERT model configuration.
        self.custom_config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.custom_config.num_hidden_layers = 4        # Fewer layers for faster training
        self.custom_config.dim = 768                    # Keeping hidden size same as BERT-base
        self.custom_config.ffn_dim = 3072               # Feed-forward network size (usually 4x hidden size)
        self.custom_config.n_heads = 12                 # Number of attention heads
        self.custom_config.dropout = 0.2                # Increase dropout to 20%
        self.custom_config.attention_dropout = 0.2      # Increase attention dropout
        # Load pre-trained DistilBERT model with costomized configuration.
        self.distilbert = DistilBertModel(self.custom_config)
        # Resize token embeddings if we've added new tokens (special markers)
        self.distilbert.resize_token_embeddings(len(tokenizer))

        # Define a dropout layer for regularization.
        self.dropout = nn.Dropout(self.custom_config.seq_classif_dropout)
        # Define a classification head on top of the pooled output.
        self.classifier = nn.Linear(self.custom_config.dim, num_labels)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tensor of token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens are padding.
        
        Returns:
            logits (torch.Tensor): Predicted logits for each sentiment class.
        """
        # Obtain the transformer outputs.
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT does not have a pooled output by default, so we take the representation of the first token.
        hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_len, hidden_dim)
        pooled_output = hidden_state[:, 0]  # Use the representation of the first token (which is analogous to [CLS])
        
        # Apply dropout.
        pooled_output = self.dropout(pooled_output)
        # Pass through the classifier layer to obtain logits.
        logits = self.classifier(pooled_output)  # shape: (batch_size, num_labels)
        
        return logits
    
class TransformerTrainer:
    def __init__(self, model, train_dataset, val_dataset, device,
                num_epochs=5, batch_size=32, learning_rate=2e-5,
                max_grad_norm=1.0, patience=2, restore = True,
                save_path = "DistilBert.pth"):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=True)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.restore = restore
        self.save_path = save_path

        self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        total_steps = len(self.train_loader) * num_epochs

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    def train(self):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss = self.train_one_epoch()
            val_loss = self.validate()

            print(f"Average Train Loss in epoch {epoch}/{self.num_epochs}: {train_loss:.4f}", 
                f"\nAverage Validation Loss in epoch {epoch}/{self.num_epochs}: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
            else:
                patience_counter += 1
            
                if patience_counter >= self.patience:
                    if self.restore:
                        self.model.load_state_dict(torch.load(self.save_path))
                        print(f"Early stopping at epoch {epoch}. "
                            f"Restore model at epoch {epoch-self.patience}.")
                    else: 
                        torch.save(self.model.state_dict(), "DistilBert.pth")
                        print(f"Early stopping at epoch {epoch}.")
                    break
        return self.model

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in loop:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # LogSoftmax implemented inside nn.CrossEntropyLoss() converts logits to probabilities.
            loss = self.criterion(logits, labels)
            loss.backward()

            # Clip gradients to prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            # Update parameters.
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        return total_loss/len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

        return total_loss/len(self.val_loader)
    
def inference(model, device, dataset, batch_size):
    model.to(device)
    model.eval()

    test_loader = DataLoader(
            dataset, batch_size, shuffle=False)
    
    predictions = []
    probabilities = []
    labels = []
    label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
    reverse_label_map = {v: k for k, v in label_map.items()}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(logits, tuple):  # If model returns a tuple (e.g. for some heads)
                logits = logits[0]
            
            # Use SoftMax function to  convert raw value to probabilities.
            probs = F.softmax(logits, dim=1)

            # Select the position with highest probability.
            preds = torch.argmax(probs, dim=1)

            # Convert results to list and 
            predictions.extend(preds.tolist())
            probabilities.extend(probs.tolist())
            labels.extend(batch['label'])

    predictions_text = [reverse_label_map[label] for label in predictions]
    accuracy = sum(p == t for p, t in zip(predictions, labels))/len(predictions)

    print(f'The prediction accuracy is {accuracy}')

    return probabilities, predictions, predictions_text