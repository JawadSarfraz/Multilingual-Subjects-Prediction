import json
import os
from typing import Dict, List, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from model import SubjectPredictor

class AbstractDataset(Dataset):
    def __init__(self, texts: List[str], subjects: List[List[str]]):
        self.texts = texts
        self.subjects = subjects
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[str, List[str]]:
        return self.texts[idx], self.subjects[idx]

class Trainer:
    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        device: str = None,
        model_dir: str = "../../models"
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.model_dir = model_dir
        self.predictor = SubjectPredictor(model_name, max_length, device)
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def _create_dataloader(
        self,
        texts: List[str],
        subjects: List[List[str]],
        shuffle: bool = True
    ) -> DataLoader:
        dataset = AbstractDataset(texts, subjects)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Tuple[str, List[str]]]) -> Dict:
        texts, subjects = zip(*batch)
        return self.predictor.prepare_features(list(texts), list(subjects))
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        return {
            'f1_micro': f1_score(labels, predictions, average='micro'),
            'f1_macro': f1_score(labels, predictions, average='macro'),
            'precision_micro': precision_score(labels, predictions, average='micro'),
            'precision_macro': precision_score(labels, predictions, average='macro'),
            'recall_micro': recall_score(labels, predictions, average='micro'),
            'recall_macro': recall_score(labels, predictions, average='macro')
        }
    
    def train(
        self,
        train_data: Dict,
        val_data: Dict,
        model_name: str = "subject_classifier"
    ) -> Dict[str, List[float]]:
        # Initialize training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        # Create model
        unique_subjects = set()
        for subjects in train_data['subjects']:
            unique_subjects.update(subjects)
        num_labels = len(unique_subjects)
        self.predictor.create_model(num_labels)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.predictor.model.parameters(),
            lr=self.learning_rate
        )
        
        # Create dataloaders
        train_dataloader = self._create_dataloader(
            train_data['abstract'],
            train_data['subjects']
        )
        val_dataloader = self._create_dataloader(
            val_data['abstract'],
            val_data['subjects'],
            shuffle=False
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            # Training phase
            train_loss = 0
            self.predictor.model.train()
            progress_bar = tqdm(train_dataloader, desc="Training")
            
            for batch in progress_bar:
                loss = self.predictor.train_step(batch, optimizer)
                train_loss += loss
                progress_bar.set_postfix({'loss': loss})
            
            train_loss = train_loss / len(train_dataloader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss = 0
            all_predictions = []
            all_labels = []
            
            self.predictor.model.eval()
            for batch in tqdm(val_dataloader, desc="Validation"):
                loss, predictions = self.predictor.evaluate(batch)
                val_loss += loss
                all_predictions.append(predictions)
                all_labels.append(batch['labels'].cpu().numpy())
            
            val_loss = val_loss / len(val_dataloader)
            history['val_loss'].append(val_loss)
            
            # Compute validation metrics
            predictions = np.vstack(all_predictions)
            labels = np.vstack(all_labels)
            metrics = self._compute_metrics(predictions, labels)
            history['val_metrics'].append(metrics)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("Validation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(self.model_dir, f"{model_name}_best.pt")
                self.predictor.save_model(model_path)
                print(f"Saved best model to {model_path}")
        
        return history
    
    def evaluate_model(
        self,
        test_data: Dict,
        model_path: str = None
    ) -> Tuple[float, Dict[str, float]]:
        # Load best model if path provided
        if model_path:
            self.predictor.load_model(model_path)
        
        # Create test dataloader
        test_dataloader = self._create_dataloader(
            test_data['abstract'],
            test_data['subjects'],
            shuffle=False
        )
        
        # Evaluation
        test_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in tqdm(test_dataloader, desc="Testing"):
            loss, predictions = self.predictor.evaluate(batch)
            test_loss += loss
            all_predictions.append(predictions)
            all_labels.append(batch['labels'].cpu().numpy())
        
        test_loss = test_loss / len(test_dataloader)
        
        # Compute metrics
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        metrics = self._compute_metrics(predictions, labels)
        
        return test_loss, metrics

def load_data(data_path: str) -> Dict:
    """Load processed data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {
        'abstract': [doc['abstract'] for doc in data],
        'subjects': [doc['subjects'] for doc in data]
    }

if __name__ == "__main__":
    # Load data
    data_dir = "../../data/processed"
    train_data = load_data(os.path.join(data_dir, "train.json"))
    val_data = load_data(os.path.join(data_dir, "val.json"))
    test_data = load_data(os.path.join(data_dir, "test.json"))
    
    # Initialize trainer
    trainer = Trainer()
    
    # Train model
    print("Starting training...")
    history = trainer.train(train_data, val_data)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.evaluate_model(
        test_data,
        os.path.join(trainer.model_dir, "subject_classifier_best.pt")
    )
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print("Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}") 