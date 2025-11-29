"""
Unit tests for data loading functionality
"""
import pytest
import pandas as pd
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from load_data import download_dataset


class TestDataLoading:
    """Test data loading functions"""
    
    def test_dataset_exists(self):
        """Test that the dataset file exists"""
        dataset_path = "data/sms_spam.csv"
        assert os.path.exists(dataset_path), f"Dataset not found at {dataset_path}"
    
    def test_dataset_structure(self):
        """Test that dataset has correct structure"""
        df = pd.read_csv("data/sms_spam.csv")
        
        # Check columns exist
        assert 'label' in df.columns, "Missing 'label' column"
        assert 'text' in df.columns, "Missing 'text' column"
        
        # Check data types
        assert df['label'].dtype == 'object', "Label should be object type"
        assert df['text'].dtype == 'object', "Text should be object type"
    
    def test_dataset_not_empty(self):
        """Test that dataset contains data"""
        df = pd.read_csv("data/sms_spam.csv")
        assert len(df) > 0, "Dataset is empty"
        # Allow for slight variation in dataset size (5570-5580 rows)
        assert 5500 <= len(df) <= 5600, f"Expected ~5574 rows, got {len(df)}"
    
    def test_labels_valid(self):
        """Test that labels are valid (ham or spam)"""
        df = pd.read_csv("data/sms_spam.csv")
        valid_labels = {'ham', 'spam'}
        unique_labels = set(df['label'].unique())
        assert unique_labels.issubset(valid_labels), f"Invalid labels found: {unique_labels - valid_labels}"
    
    def test_no_missing_values(self):
        """Test that there are no missing values in critical columns"""
        df = pd.read_csv("data/sms_spam.csv")
        assert df['label'].isnull().sum() == 0, "Found missing values in label column"
        assert df['text'].isnull().sum() == 0, "Found missing values in text column"
    
    def test_class_distribution(self):
        """Test that both classes are present"""
        df = pd.read_csv("data/sms_spam.csv")
        class_counts = df['label'].value_counts()
        
        assert 'ham' in class_counts.index, "Ham class not found"
        assert 'spam' in class_counts.index, "Spam class not found"
        assert class_counts['ham'] > 0, "No ham samples"
        assert class_counts['spam'] > 0, "No spam samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
