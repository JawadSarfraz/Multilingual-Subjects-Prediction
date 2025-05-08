from langdetect import detect
from typing import Dict, Union, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class LanguageHandler:
    def __init__(self):
        # Initialize translation model and tokenizer
        self.model_name = 'Helsinki-NLP/opus-mt-de-en'
        self.tokenizer = None
        self.model = None
    
    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        try:
            return detect(text)
        except:
            return 'unknown'
    
    def load_translation_model(self):
        """Load the translation model and tokenizer."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
    
    def translate_to_english(self, text: str) -> str:
        """Translate German text to English."""
        self.load_translation_model()
        
        # Tokenize and translate
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        
        # Decode the translation
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    def process_multilingual_text(self, text: Union[str, List[str]]) -> str:
        """Process text that could be in different languages."""
        # Handle list of text
        if isinstance(text, list):
            text = ' '.join(text)
        
        if not text:
            return ""
            
        # Detect language
        lang = self.detect_language(text)
        
        # Translate if German
        if lang == 'de':
            return self.translate_to_english(text)
        
        return text

def process_multilingual_document(doc: Dict) -> Dict:
    """Process a document that might contain multilingual content."""
    handler = LanguageHandler()
    
    processed_doc = doc.copy()
    
    # Process abstract
    if 'abstract' in doc:
        processed_doc['abstract'] = handler.process_multilingual_text(doc['abstract'])
    
    # Process subjects
    if 'subject' in doc:
        processed_subjects = []
        for subject in doc['subject']:
            processed_subject = handler.process_multilingual_text(subject)
            if processed_subject:
                processed_subjects.append(processed_subject)
        processed_doc['subject'] = processed_subjects
    
    return processed_doc
