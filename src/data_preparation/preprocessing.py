import re
import nltk
from typing import List, Dict, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english') + stopwords.words('german'))
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text."""
        words = word_tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def preprocess_abstract(self, abstract: Union[str, List[str]]) -> str:
        """Preprocess abstract text."""
        # Handle list of abstracts
        if isinstance(abstract, list):
            abstract = ' '.join(abstract)
            
        # Apply preprocessing steps
        text = self.clean_text(abstract)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        
        return text
    
    def preprocess_subjects(self, subjects: List[str]) -> List[str]:
        """Preprocess subject labels."""
        processed_subjects = []
        for subject in subjects:
            # Clean and normalize subject
            subject = self.clean_text(subject)
            subject = self.lemmatize_text(subject)
            if subject:  # Only add non-empty subjects
                processed_subjects.append(subject)
        
        return list(set(processed_subjects))  # Remove duplicates

def process_document(doc: Dict) -> Dict:
    """Process a single document."""
    preprocessor = TextPreprocessor()
    
    processed_doc = {
        'id': doc.get('econbiz_id', ''),
        'abstract': preprocessor.preprocess_abstract(doc.get('abstract', '')),
        'subjects': preprocessor.preprocess_subjects(doc.get('subject', [])),
        'language': doc.get('language', [''])[0],
        'title': doc.get('title', ''),
    }
    
    return processed_doc
