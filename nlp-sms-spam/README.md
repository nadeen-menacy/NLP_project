# SMS Spam Detection using NLP & Neural Networks

**University of Debrecen – Department of Data Science and Visualization**  
**Students:** Nadine Adel Menacy (A9ATIE), Aboumoussa Mohamed Aly (JE1WAQ), Sahriar Rahman Nahid (IBRO2O)  
**Course:** Introduction to Natural Language Text Processing (Róbert Lakatos, 2024/25) | **Semester:** Autumn 2025

## Overview

Automated SMS classification system using NLP and ML to detect spam messages. Includes comprehensive EDA, 4 ML models (Naive Bayes, Logistic Regression, Linear SVM, BiLSTM), extensive evaluation, unit tests, and an interactive Streamlit web app.

**Key Features:** EDA with visualizations | Multiple ML models | ROC curves & confusion matrices | Unit tests | Web application | Jupyter notebook

## Project Structure

```
nlp-sms-spam/
├── README.md, REPORT.md, requirements.txt
├── data/sms_spam.csv                    # 5,574 SMS messages
├── src/                                 # Preprocessing, models, training
├── models/                              # Trained models (.joblib, .h5)
├── notebooks/                           # Complete analysis notebook
├── app/streamlit_app.py                 # Web application
└── tests/                               # Unit tests (32 tests)
```

## Pipeline

**Data (5,574 SMS)** → **EDA** (class distribution, word clouds, bigrams) → **Preprocessing** (lowercase, URL/email/number normalization, punctuation removal) → **Features** (TF-IDF 5000, bigrams) → **Models** (NB, LR, SVM, BiLSTM) → **Evaluation** (confusion matrices, ROC, metrics) → **Deployment** (Streamlit app)

## Dataset

- **Source:** [UCI ML Repository SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Size:** 5,574 messages | **Ham:** 4,827 (86.6%) | **Spam:** 747 (13.4%) | **Imbalance:** 6.5:1

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|------------|
| **Linear SVM** | **98.2%** | **98.9%** | **95.3%** | **97.1%** | **0.991** | 0.05s |
| Logistic Regression | 97.9% | 98.4% | 94.8% | 96.6% | 0.989 | 0.12s |
| BiLSTM | 97.7% | 98.1% | 94.5% | 96.3% | 0.987 | ~30s |
| Naive Bayes | 97.4% | 97.6% | 94.1% | 95.8% | 0.982 | 0.02s |

**Winner:** Linear SVM - Best accuracy, fast training, high precision

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data (optional - dataset already included)
python src/load_data.py             # Downloads if data/sms_spam.csv missing

# 3. Train models (optional - trained models included)
python src/run_training.py          # Classical models (fast: ~1s)
python src/train_bilstm.py          # Neural network (CPU mode, ~60s)

# 4. Run tests
CUDA_VISIBLE_DEVICES='-1' pytest tests/ -v    # All 31 tests (CPU mode)

# 5. Launch web app
CUDA_VISIBLE_DEVICES='-1' streamlit run app/streamlit_app.py  # http://localhost:8501

# 6. View analysis notebook
jupyter notebook notebooks/SMS_Spam_Detection_Analysis.ipynb
```

## Models & Technologies

**Classical ML:** Naive Bayes, Logistic Regression, Linear SVM (scikit-learn)  
**Neural Network:** BiLSTM with Embedding layer (TensorFlow/Keras)  
**Features:** TF-IDF vectorization (5000 features, bigrams) + Tokenization (BiLSTM)  
**Tools:** Python 3.13, pandas, numpy, matplotlib, seaborn, wordcloud, streamlit, pytest

## Notebook Content

1. Data exploration & statistics
2. EDA: class distribution, message length, word clouds, top words, bigrams
3. Text preprocessing pipeline
4. TF-IDF & tokenization
5. Train all 4 models
6. Confusion matrices, ROC curves, performance charts
7. Conclusions & future work

## Unit Tests (32 Tests)

**test_data_loading.py:** Dataset exists, structure, no missing values, valid labels  
**test_preprocessing.py:** Text cleaning, URL/email/number replacement, TF-IDF  
**test_models.py:** Models load, predictions work, accuracy >90%, consistent results

## Limitations

1. **Class imbalance** (87% ham, 13% spam)
2. **Small dataset** (5,574 messages)
3. **English-only**
4. **Temporal drift** - spam evolves
5. **Short context** - limited information
6. **Basic features** - TF-IDF only

## Future Work

**Models:** BERT/RoBERTa, ensemble methods, transformers  
**Features:** Metadata (length, punctuation), phone/URL patterns, time-based  
**Data:** SMOTE, back-translation, synthetic generation  
**Deployment:** Confidence scores, feedback loop, REST API, cloud hosting  
**Explainability:** LIME/SHAP, feature importance, attention visualization  
**Optimization:** Quantization, batch processing, caching, multi-threading

## Academic Context

**Course:** Introduction to Natural Language Text Processing  
**Institution:** University of Debrecen, Dept. of Data Science & Visualization  
**Instructor:** Róbert Lakatos | **Semester:** Autumn 2025

**Learning Objectives:** Text preprocessing | Feature extraction | ML classification | Neural networks | Evaluation | Deployment

## Team

**Nadine Adel Menacy** (A9ATIE) | **Aboumoussa Mohamed Aly** (JE1WAQ) | **Sahriar Rahman Nahid** (IBRO2O)

## Conclusion

Complete production-ready NLP pipeline achieving **98.2% accuracy** with Linear SVM. Comprehensive EDA, multiple models, rigorous testing, and interactive deployment showcase ML engineering best practices.

**Documentation:** README.md (this file) | REPORT.md (technical details) | Jupyter notebook (complete analysis)