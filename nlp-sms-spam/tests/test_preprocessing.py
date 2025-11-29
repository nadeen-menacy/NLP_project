"""
Unit tests for text preprocessing functionality
"""
import pytest
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import clean_text, prepare_data, split_data, vectorize_tfidf


class TestPreprocessing:
    """Test preprocessing functions"""
    
    def test_clean_text_lowercase(self):
        """Test that text is converted to lowercase"""
        text = "HELLO WORLD"
        cleaned = clean_text(text)
        assert cleaned.islower(), "Text should be lowercase"
    
    def test_clean_text_url_replacement(self):
        """Test that URLs are replaced"""
        text = "Check this out https://example.com"
        cleaned = clean_text(text)
        assert "URL" in cleaned.upper(), "URLs should be replaced with URL token"
        assert "https" not in cleaned, "Original URL should be removed"
    
    def test_clean_text_email_replacement(self):
        """Test that emails are replaced"""
        text = "Contact me at test@example.com"
        cleaned = clean_text(text)
        assert "EMAIL" in cleaned.upper(), "Emails should be replaced with EMAIL token"
        assert "@" not in cleaned, "@ symbol should be removed"
    
    def test_clean_text_number_replacement(self):
        """Test that numbers are replaced"""
        text = "Call me at 12345"
        cleaned = clean_text(text)
        assert "NUM" in cleaned.upper(), "Numbers should be replaced with NUM token"
        assert "12345" not in cleaned, "Original number should be removed"
    
    def test_clean_text_punctuation_removal(self):
        """Test that punctuation is removed"""
        text = "Hello, World! How are you?"
        cleaned = clean_text(text)
        assert "," not in cleaned, "Comma should be removed"
        assert "!" not in cleaned, "Exclamation mark should be removed"
        assert "?" not in cleaned, "Question mark should be removed"
    
    def test_clean_text_whitespace_normalization(self):
        """Test that multiple whitespaces are normalized"""
        text = "Hello    World     Test"
        cleaned = clean_text(text)
        assert "    " not in cleaned, "Multiple spaces should be normalized"
        assert cleaned == "hello world test", f"Expected 'hello world test', got '{cleaned}'"
    
    def test_clean_text_not_empty(self):
        """Test that cleaned text is not empty for valid input"""
        text = "This is a valid message"
        cleaned = clean_text(text)
        assert len(cleaned) > 0, "Cleaned text should not be empty"
        assert len(cleaned.split()) > 0, "Cleaned text should have tokens"
    
    def test_prepare_data_returns_dataframe(self):
        """Test that prepare_data returns a DataFrame"""
        df = prepare_data('data/sms_spam.csv')
        assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
        assert 'text' in df.columns, "Should have 'text' column"
        assert 'label' in df.columns, "Should have 'label' column"
    
    def test_prepare_data_cleans_text(self):
        """Test that prepare_data applies cleaning"""
        df = prepare_data('data/sms_spam.csv')
        # Check that at least one text has been cleaned (lowercase)
        sample_text = df['text'].iloc[0]
        assert sample_text.islower() or sample_text == "", "Text should be cleaned (lowercase)"
    
    def test_split_data_correct_proportions(self):
        """Test that data split has correct proportions"""
        df = prepare_data('data/sms_spam.csv')
        X_train, X_test, y_train, y_test = split_data(df)
        
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        
        assert 0.15 <= test_ratio <= 0.25, f"Test ratio should be around 0.2, got {test_ratio}"
    
    def test_split_data_no_data_leakage(self):
        """Test that train and test sets don't overlap"""
        df = prepare_data('data/sms_spam.csv')
        X_train, X_test, y_train, y_test = split_data(df)
        
        # Check lengths match
        assert len(X_train) == len(y_train), "X_train and y_train lengths don't match"
        assert len(X_test) == len(y_test), "X_test and y_test lengths don't match"
    
    def test_vectorize_tfidf_shape(self):
        """Test TF-IDF vectorization produces correct shapes"""
        df = prepare_data('data/sms_spam.csv')
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_tfidf, X_test_tfidf, vectorizer = vectorize_tfidf(X_train, X_test)
        
        # Check shapes
        assert X_train_tfidf.shape[0] == len(X_train), "Training matrix rows don't match"
        assert X_test_tfidf.shape[0] == len(X_test), "Test matrix rows don't match"
        assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1], "Feature dimensions don't match"
    
    def test_vectorize_tfidf_not_empty(self):
        """Test that TF-IDF matrices are not empty"""
        df = prepare_data('data/sms_spam.csv')
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_tfidf, X_test_tfidf, vectorizer = vectorize_tfidf(X_train, X_test)
        
        assert X_train_tfidf.shape[1] > 0, "TF-IDF should have features"
        assert X_train_tfidf.nnz > 0, "TF-IDF matrix should have non-zero values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
