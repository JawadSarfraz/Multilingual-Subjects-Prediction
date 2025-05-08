import json
import os
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
from preprocessing import TextPreprocessor, process_document
from language_detection import LanguageHandler, process_multilingual_document
from subject_standardization import SubjectStandardizer, standardize_document_subjects

class DataProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        print(f"Initializing DataProcessor with raw_data_path: {raw_data_path}")
        print(f"processed_data_path: {processed_data_path}")
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.language_handler = LanguageHandler()
        self.text_preprocessor = TextPreprocessor()
        self.subject_standardizer = SubjectStandardizer()
        
    def load_data(self) -> List[Dict]:
        """Load raw data from JSON file."""
        print(f"Loading data from {self.raw_data_path}")
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} documents")
        return data
    
    def save_data(self, data: List[Dict], filename: str):
        """Save processed data to JSON file."""
        output_path = os.path.join(self.processed_data_path, filename)
        print(f"Saving data to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} documents to {filename}")
    
    def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process all documents in the dataset."""
        processed_data = []
        
        # First pass: collect all subjects for standardization
        print("Collecting subjects for standardization...")
        all_subjects = []
        for doc in data:
            if 'subject' in doc:
                all_subjects.extend(doc['subject'])
        print(f"Found {len(all_subjects)} total subjects")
        
        # Fit subject standardizer
        print("Fitting subject standardizer...")
        self.subject_standardizer.fit(all_subjects)
        
        # Process each document
        print("Processing documents...")
        for doc in tqdm(data, desc="Processing documents"):
            # Step 1: Handle multilingual content
            doc = process_multilingual_document(doc)
            
            # Step 2: Clean and preprocess text
            doc = process_document(doc)
            
            # Step 3: Standardize subjects
            doc = standardize_document_subjects(doc, self.subject_standardizer)
            
            if doc['abstract'] and doc['subjects']:  # Only keep documents with both abstract and subjects
                processed_data.append(doc)
        
        print(f"Processed {len(processed_data)} documents")
        return processed_data
    
    def split_data(self, data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
        """Split data into train, validation, and test sets."""
        print("Splitting data...")
        # Convert to DataFrame for easier splitting
        df = pd.DataFrame(data)
        
        # Calculate split sizes
        n = len(df)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        train_data = df[:train_size].to_dict('records')
        val_data = df[train_size:train_size+val_size].to_dict('records')
        test_data = df[train_size+val_size:].to_dict('records')
        
        print(f"Split sizes: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        return train_data, val_data, test_data
    
    def run_pipeline(self):
        """Run the complete data processing pipeline."""
        print("\nStarting data processing pipeline...")
        
        # Create output directory if it doesn't exist
        print(f"Creating output directory: {self.processed_data_path}")
        os.makedirs(self.processed_data_path, exist_ok=True)
        
        # Load data
        print("\nLoading data...")
        data = self.load_data()
        
        # Process data
        print("\nProcessing data...")
        processed_data = self.process_data(data)
        
        # Split data
        print("\nSplitting data...")
        train_data, val_data, test_data = self.split_data(processed_data)
        
        # Save processed datasets
        print("\nSaving processed data...")
        self.save_data(train_data, 'train.json')
        self.save_data(val_data, 'val.json')
        self.save_data(test_data, 'test.json')
        
        print("\nData processing complete!")
        print(f"Train set size: {len(train_data)}")
        print(f"Validation set size: {len(val_data)}")
        print(f"Test set size: {len(test_data)}")

if __name__ == "__main__":
    print("Starting script...")
    # Set up paths
    raw_data_path = "../../data/raw/sample_data_500.json"
    processed_data_path = "../../data/processed"
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Raw data path exists: {os.path.exists(raw_data_path)}")
    print(f"Absolute raw data path: {os.path.abspath(raw_data_path)}")
    
    # Initialize and run processor
    processor = DataProcessor(raw_data_path, processed_data_path)
    processor.run_pipeline()
