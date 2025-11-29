"""
Unit tests for model functionality
"""
import pytest
import joblib
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from preprocess import prepare_data, split_data, vectorize_tfidf


class TestModels:
    """Test model loading and prediction"""
    
    def test_models_exist(self):
        """Test that all model files exist"""
        models = ['NaiveBayes', 'LogisticRegression', 'LinearSVC']
        for model_name in models:
            model_path = f"models/{model_name}.joblib"
            assert os.path.exists(model_path), f"Model file not found: {model_path}"
    
    def test_vectorizer_exists(self):
        """Test that TF-IDF vectorizer exists"""
        vectorizer_path = "models/tfidf_vectorizer.joblib"
        assert os.path.exists(vectorizer_path), f"Vectorizer not found: {vectorizer_path}"
    
    def test_bilstm_model_exists(self):
        """Test that BiLSTM model exists"""
        model_path = "models/bilstm_model.h5"
        tokenizer_path = "models/lstm_tokenizer.joblib"
        
        assert os.path.exists(model_path), f"BiLSTM model not found: {model_path}"
        assert os.path.exists(tokenizer_path), f"LSTM tokenizer not found: {tokenizer_path}"
    
    def test_load_classical_models(self):
        """Test that classical models can be loaded"""
        models = ['NaiveBayes', 'LogisticRegression', 'LinearSVC']
        for model_name in models:
            model = joblib.load(f"models/{model_name}.joblib")
            assert model is not None, f"Failed to load {model_name}"
            assert hasattr(model, 'predict'), f"{model_name} should have predict method"
    
    def test_load_vectorizer(self):
        """Test that vectorizer can be loaded"""
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        assert vectorizer is not None, "Failed to load vectorizer"
        assert hasattr(vectorizer, 'transform'), "Vectorizer should have transform method"
    
    def test_model_predictions_classical(self):
        """Test that classical models can make predictions"""
        # Load vectorizer and a model
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        model = joblib.load("models/NaiveBayes.joblib")
        
        # Test messages
        test_messages = [
            "Hello, how are you?",
            "URGENT! You have won $1000! Call now!",
            "Can we meet tomorrow for lunch?"
        ]
        
        # Transform and predict
        X = vectorizer.transform(test_messages)
        predictions = model.predict(X)
        
        # Verify predictions
        assert len(predictions) == len(test_messages), "Should predict for all messages"
        assert all(pred in ['ham', 'spam'] for pred in predictions), "Predictions should be ham or spam"
    
    def test_model_prediction_shapes(self):
        """Test that predictions have correct shape"""
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        model = joblib.load("models/LogisticRegression.joblib")
        
        test_message = ["This is a test message"]
        X = vectorizer.transform(test_message)
        predictions = model.predict(X)
        
        assert predictions.shape[0] == 1, "Should predict for single message"
    
    def test_bilstm_prediction(self):
        """Test that BiLSTM model can make predictions"""
        try:
            import os
            # Force TensorFlow to use CPU only to avoid GPU/CUDA issues
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
        except ImportError:
            pytest.skip("TensorFlow not installed")
        
        try:
            # Load model and tokenizer
            model = load_model("models/bilstm_model.h5")
            tokenizer = joblib.load("models/lstm_tokenizer.joblib")
            
            # Test message
            test_message = ["Hello, how are you?"]
            
            # Preprocess
            sequences = tokenizer.texts_to_sequences(test_message)
            padded = pad_sequences(sequences, maxlen=100, padding="post")
            
            # Predict (verbose=0 to suppress output)
            prediction = model.predict(padded, verbose=0)
            
            assert prediction is not None, "BiLSTM should return predictions"
            assert prediction.shape[0] == 1, "Should predict for single message"
            assert 0 <= prediction[0][0] <= 1, "BiLSTM output should be probability [0,1]"
        except Exception as e:
            # If there are GPU/CUDA issues, skip the test gracefully
            if "GPU" in str(e) or "CUDA" in str(e) or "JIT compilation failed" in str(e):
                pytest.skip(f"GPU/CUDA configuration issue: {str(e)}")
            else:
                raise
    
    def test_model_consistency(self):
        """Test that model predictions are consistent"""
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        model = joblib.load("models/LinearSVC.joblib")
        
        test_message = ["Free prize! Call now to claim your reward!"]
        X = vectorizer.transform(test_message)
        
        # Predict multiple times
        pred1 = model.predict(X)[0]
        pred2 = model.predict(X)[0]
        pred3 = model.predict(X)[0]
        
        assert pred1 == pred2 == pred3, "Model predictions should be consistent"
    
    def test_models_on_spam_example(self):
        """Test that models correctly classify obvious spam"""
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        
        spam_message = ["WINNER! You have won $1000000! Click here NOW! Free prize! Urgent!"]
        X = vectorizer.transform(spam_message)
        
        models = ['NaiveBayes', 'LogisticRegression', 'LinearSVC']
        predictions = []
        
        for model_name in models:
            model = joblib.load(f"models/{model_name}.joblib")
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        # At least 2 out of 3 models should classify as spam
        spam_count = sum(1 for p in predictions if p == 'spam')
        assert spam_count >= 2, f"Most models should classify obvious spam correctly, got {predictions}"
    
    def test_models_on_ham_example(self):
        """Test that models correctly classify obvious ham"""
        vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
        
        ham_message = ["Hi, are you free for dinner tonight? Let me know."]
        X = vectorizer.transform(ham_message)
        
        models = ['NaiveBayes', 'LogisticRegression', 'LinearSVC']
        predictions = []
        
        for model_name in models:
            model = joblib.load(f"models/{model_name}.joblib")
            pred = model.predict(X)[0]
            predictions.append(pred)
        
        # At least 2 out of 3 models should classify as ham
        ham_count = sum(1 for p in predictions if p == 'ham')
        assert ham_count >= 2, f"Most models should classify obvious ham correctly, got {predictions}"


class TestModelPerformance:
    """Test model performance metrics"""
    
    def test_models_accuracy_threshold(self):
        """Test that models meet minimum accuracy threshold"""
        from sklearn.metrics import accuracy_score
        
        # Prepare test data
        df = prepare_data('data/sms_spam.csv')
        X_train, X_test, y_train, y_test = split_data(df)
        X_train_tfidf, X_test_tfidf, _ = vectorize_tfidf(X_train, X_test)
        
        models = ['NaiveBayes', 'LogisticRegression', 'LinearSVC']
        min_accuracy = 0.90  # 90% minimum threshold
        
        for model_name in models:
            model = joblib.load(f"models/{model_name}.joblib")
            predictions = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, predictions)
            
            assert accuracy >= min_accuracy, f"{model_name} accuracy {accuracy:.4f} below threshold {min_accuracy}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
