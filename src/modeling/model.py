import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Union, Tuple
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

class SubjectClassifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use CLS token output for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

class SubjectPredictor:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.model = None
        self.label_binarizer = MultiLabelBinarizer()
        
    def prepare_features(
        self,
        texts: List[str],
        labels: List[List[str]] = None
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        # Tokenize texts
        features = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Convert to device
        features = {k: v.to(self.device) for k, v in features.items()}
        
        if labels is not None:
            # Fit and transform labels if not fitted
            if not hasattr(self.label_binarizer, 'classes_'):
                y = self.label_binarizer.fit_transform(labels)
            else:
                y = self.label_binarizer.transform(labels)
            features['labels'] = torch.FloatTensor(y).to(self.device)
        
        return features
    
    def create_model(self, num_labels: int) -> None:
        """Initialize the model."""
        self.model = SubjectClassifier(
            model_name=self.tokenizer.name_or_path,
            num_labels=num_labels
        ).to(self.device)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Perform one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        loss, _ = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, features: Dict[str, torch.Tensor]) -> Tuple[float, np.ndarray]:
        """Evaluate the model on validation/test data."""
        self.model.eval()
        
        with torch.no_grad():
            loss, logits = self.model(
                input_ids=features['input_ids'],
                attention_mask=features['attention_mask'],
                labels=features['labels']
            )
            
            predictions = torch.sigmoid(logits).cpu().numpy()
            predictions = (predictions > 0.5).astype(int)
        
        return loss.item(), predictions
    
    def predict(self, texts: List[str]) -> List[List[str]]:
        """Predict subjects for new texts."""
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model first.")
        
        self.model.eval()
        features = self.prepare_features(texts)
        
        with torch.no_grad():
            logits = self.model(
                input_ids=features['input_ids'],
                attention_mask=features['attention_mask']
            )
            predictions = torch.sigmoid(logits).cpu().numpy()
            predictions = (predictions > 0.5).astype(int)
        
        return self.label_binarizer.inverse_transform(predictions)
    
    def save_model(self, path: str) -> None:
        """Save model and label binarizer."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model state and config
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer_name': self.tokenizer.name_or_path,
            'max_length': self.max_length,
            'label_classes': self.label_binarizer.classes_
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load saved model and label binarizer."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Initialize label binarizer with saved classes
        self.label_binarizer.fit([checkpoint['label_classes']])
        
        # Create and load model
        self.create_model(len(checkpoint['label_classes']))
        self.model.load_state_dict(checkpoint['model_state_dict']) 