import spacy
import textstat
from collections import Counter
import re
from typing import Dict, Tuple, Optional
import pandas as pd
from datetime import datetime
import csv
from pathlib import Path
from textblob import TextBlob
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('text_analysis.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class TextAnalyzer:
    def __init__(self):
        try:
            self.nlp = spacy.load('en_core_web_sm')
            logging.info("Successfully initialized spaCy model")
        except OSError as e:
            logging.error(f"Failed to load spaCy model: {e}")
            raise Exception("Please install spaCy model using: python -m spacy download en_core_web_sm")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess the input text."""
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        
        try:
            # Remove extra whitespace and normalize
            text = re.sub(r'\s+', ' ', text).strip()
            # Remove special characters but keep punctuation
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            return text
        except Exception as e:
            logging.error(f"Error in preprocessing text: {e}")
            raise

    def calculate_lexical_diversity(self, text: str) -> float:
        """Calculate type-token ratio for lexical diversity."""
        words = text.split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def calculate_complexity_metrics(self, text: str) -> Dict[str, float]:
        """Calculate various complexity metrics for the text."""
        try:
            metrics = {
                'flesch_reading_ease': textstat.flesch_reading_ease(text),
                'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
                'gunning_fog': textstat.gunning_fog(text),
                'smog_index': textstat.smog_index(text),
                'automated_readability_index': textstat.automated_readability_index(text),
                'coleman_liau_index': textstat.coleman_liau_index(text),
                'linsear_write': textstat.linsear_write_formula(text),
                'dale_chall': textstat.dale_chall_readability_score(text),
                'lexical_diversity': self.calculate_lexical_diversity(text),
                'average_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0,
                'sentence_count': textstat.sentence_count(text),
                'syllable_count': textstat.syllable_count(text),
            }
            
            # Add spaCy-based metrics
            doc = self.nlp(text)
            
            # Calculate sentence complexity
            sentence_lengths = [len([token for token in sent]) for sent in doc.sents]
            metrics['average_sentence_length'] = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
            
            # Calculate vocabulary sophistication
            pos_counts = Counter([token.pos_ for token in doc])
            metrics['noun_verb_ratio'] = pos_counts['NOUN'] / pos_counts['VERB'] if pos_counts['VERB'] > 0 else 0
            metrics['adjective_adverb_ratio'] = pos_counts['ADJ'] / pos_counts['ADV'] if pos_counts['ADV'] > 0 else 0
            
            # Calculate syntax complexity
            metrics['dependency_distance'] = sum(abs(token.i - token.head.i) for token in doc) / len(doc)
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error in calculating complexity metrics: {e}")
            raise

    def analyze_formality(self, text: str) -> Dict[str, float]:
        """Analyze the formality of the text."""
        try:
            doc = self.nlp(text)
            
            # Formality indicators
            formality_metrics = {
                'personal_pronouns': sum(1 for token in doc if token.pos_ == 'PRON'),
                'contractions': len(re.findall(r"'", text)) - (len(re.findall(r"'s", text)) + len(re.findall(r"s'", text))),
                'passive_voice': sum(1 for token in doc if token.dep_ == 'nsubjpass'),
                'academic_words': sum(1 for token in doc if token.pos_ in ['NOUN', 'ADJ', 'ADV'] and len(token.text) > 6),
                'coordinating_conjunctions': sum(1 for token in doc if token.pos_ == 'CCONJ'),
                'subordinating_conjunctions': sum(1 for token in doc if token.pos_ == 'SCONJ'),
                'nominalizations': sum(1 for token in doc if token.pos_ == 'NOUN' and token.text.endswith(('tion', 'ment', 'ness', 'ity'))),
            }
            
            return formality_metrics
            
        except Exception as e:
            logging.error(f"Error in analyzing formality: {e}")
            raise

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of the text using TextBlob."""
        try:
            blob = TextBlob(text)
            sentiment_metrics = {
                'polarity': blob.sentiment.polarity,  # Range: -1 (negative) to 1 (positive)
                'subjectivity': blob.sentiment.subjectivity,  # Range: 0 (objective) to 1 (subjective)
                'sentence_sentiment_variance': float(pd.Series([s.sentiment.polarity for s in blob.sentences]).var()) if blob.sentences else 0,
            }
            return sentiment_metrics
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            raise

def compare_texts(thinking_output: str, final_output: str, prompt: str = None) -> Dict[str, Dict]:
    """Compare two texts and return detailed analysis."""
    try:
        analyzer = TextAnalyzer()
        
        # Preprocess texts
        thinking_output = analyzer.preprocess_text(thinking_output)
        final_output = analyzer.preprocess_text(final_output)
        
        # Analyze both texts
        thinking_metrics = analyzer.calculate_complexity_metrics(thinking_output)
        final_metrics = analyzer.calculate_complexity_metrics(final_output)
        
        thinking_formality = analyzer.analyze_formality(thinking_output)
        final_formality = analyzer.analyze_formality(final_output)
        
        thinking_sentiment = analyzer.analyze_sentiment(thinking_output)
        final_sentiment = analyzer.analyze_sentiment(final_output)
        
        # Calculate differences
        differences = {
            'complexity_differences': {
                metric: final_metrics[metric] - thinking_metrics[metric]
                for metric in thinking_metrics
            },
            'formality_differences': {
                metric: final_formality[metric] - thinking_formality[metric]
                for metric in thinking_formality
            },
            'sentiment_differences': {
                metric: final_sentiment[metric] - thinking_sentiment[metric]
                for metric in thinking_sentiment
            }
        }
        
        return {
            'thinking_analysis': {
                'complexity': thinking_metrics,
                'formality': thinking_formality,
                'sentiment': thinking_sentiment
            },
            'final_analysis': {
                'complexity': final_metrics,
                'formality': final_formality,
                'sentiment': final_sentiment
            },
            'differences': differences
        }
        
    except Exception as e:
        logging.error(f"Error in comparing texts: {e}")
        raise

def save_to_csv(results: Dict, thinking_output: str, final_output: str, prompt: str, 
                filename: str = None) -> str:
    """Save analysis results and input texts to CSV."""
    try:
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'text_analysis_{timestamp}.csv'
        
        # Flatten the nested dictionary structure
        flat_dict = {
            'timestamp': datetime.now().isoformat(),
            'prompt': prompt,
            'thinking_output': thinking_output,
            'final_output': final_output
        }
        
        # Add flattened metrics
        for analysis_type in ['thinking_analysis', 'final_analysis']:
            for metric_type in ['complexity', 'formality', 'sentiment']:
                if metric_type in results[analysis_type]:
                    for metric, value in results[analysis_type][metric_type].items():
                        flat_dict[f'{analysis_type}_{metric_type}_{metric}'] = value
        
        # Add differences
        for diff_type in results['differences']:
            for metric, value in results['differences'][diff_type].items():
                flat_dict[f'difference_{diff_type}_{metric}'] = value
        
        # Save to CSV
        df = pd.DataFrame([flat_dict])
        df.to_csv(filename, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logging.info(f"Results saved to {filename}")
        
        return filename
        
    except Exception as e:
        logging.error(f"Error saving to CSV: {e}")
        raise

def analyze_and_save(thinking_output: str, final_output: str, prompt: str = None, 
                    filename: str = None) -> Tuple[Dict, str]:
    """Convenience function to analyze texts and save results."""
    try:
        # Validate inputs
        if not thinking_output or not final_output:
            raise ValueError("Both thinking output and final output must be provided")
            
        results = compare_texts(thinking_output, final_output, prompt)
        csv_file = save_to_csv(results, thinking_output, final_output, prompt, filename)
        
        return results, csv_file
        
    except Exception as e:
        logging.error(f"Error in analyze_and_save: {e}")
        raise

if __name__ == "__main__":
    # Example usage with error handling
    try:
        thinking_text = """Your thinking output text here"""
        final_text = """Your final output text here"""
        prompt_text = """Your prompt text here"""
        
        results, csv_file = analyze_and_save(thinking_text, final_text, prompt_text)
        logging.info(f"Analysis completed successfully. Results saved to {csv_file}")
        
    except Exception as e:
        logging.error(f"Program failed: {e}")
        sys.exit(1)
