from typing import List, Dict, Set
from collections import defaultdict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SubjectStandardizer:
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        self.subject_mapping = {}
        self.vectorizer = TfidfVectorizer()
        
    def _preprocess_subject(self, subject: str) -> str:
        """Preprocess a subject string."""
        # Convert to lowercase and remove special characters
        subject = subject.lower()
        subject = re.sub(r'[^\w\s]', ' ', subject)
        # Remove extra whitespace
        subject = re.sub(r'\s+', ' ', subject).strip()
        return subject
    
    def _compute_similarity(self, subjects: List[str]) -> np.ndarray:
        """Compute similarity matrix between subjects."""
        # Fit and transform the vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(subjects)
        # Compute cosine similarity
        return cosine_similarity(tfidf_matrix)
    
    def _find_similar_subjects(self, subjects: List[str]) -> Dict[str, Set[str]]:
        """Find groups of similar subjects."""
        # Preprocess subjects
        processed_subjects = [self._preprocess_subject(s) for s in subjects]
        
        # Compute similarity matrix
        similarity_matrix = self._compute_similarity(processed_subjects)
        
        # Group similar subjects
        similar_groups = defaultdict(set)
        n_subjects = len(subjects)
        
        for i in range(n_subjects):
            group_found = False
            for j in range(i):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    # Add to existing group
                    for key in similar_groups:
                        if subjects[j] in similar_groups[key]:
                            similar_groups[key].add(subjects[i])
                            group_found = True
                            break
            
            if not group_found:
                # Create new group
                similar_groups[subjects[i]].add(subjects[i])
        
        return similar_groups
    
    def fit(self, subjects: List[str]):
        """Fit the standardizer on a list of subjects."""
        # Find similar subject groups
        similar_groups = self._find_similar_subjects(subjects)
        
        # Create mapping from each subject to its standardized form
        for standard_subject, similar_subjects in similar_groups.items():
            for subject in similar_subjects:
                self.subject_mapping[self._preprocess_subject(subject)] = standard_subject
    
    def transform(self, subjects: List[str]) -> List[str]:
        """Transform subjects to their standardized form."""
        standardized_subjects = []
        for subject in subjects:
            processed = self._preprocess_subject(subject)
            if processed in self.subject_mapping:
                standardized_subjects.append(self.subject_mapping[processed])
            else:
                standardized_subjects.append(subject)
        return list(set(standardized_subjects))  # Remove duplicates
    
    def fit_transform(self, subjects: List[str]) -> List[str]:
        """Fit and transform in one step."""
        self.fit(subjects)
        return self.transform(subjects)

def standardize_document_subjects(doc: Dict, standardizer: SubjectStandardizer) -> Dict:
    """Standardize subjects in a document."""
    if 'subject' in doc and doc['subject']:
        doc['subject'] = standardizer.transform(doc['subject'])
    return doc
