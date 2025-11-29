# SMS Spam Detection - Technical Report

**University of Debrecen – Department of Data Science and Visualization**  
**Students:** Nadine Adel Menacy (A9ATIE), Aboumoussa Mohamed Aly (JE1WAQ), Sahriar Rahman Nahid (IBRO2O)  
**Course:** Natural Language Text Processing (Róbert Lakatos) | **Date:** November 29, 2025

## Table of Contents
1. [Introduction](#1-introduction) | 2. [Methodology](#2-methodology) | 3. [Data Analysis](#3-data-analysis) | 4. [Implementation](#4-implementation) | 5. [Results](#5-results) | 6. [Discussion](#6-discussion) | 7. [Conclusions](#7-conclusions)

## 1. Introduction

SMS spam detection is critical for mobile security. This project develops an automated classification system distinguishing legitimate messages ("ham") from spam using NLP and ML techniques.

**Objectives:** (1) Compare multiple ML approaches (2) Develop preprocessing pipeline (3) Rigorous evaluation (4) Deploy web application (5) Document best practices

**Scope:** EDA, preprocessing, feature engineering (TF-IDF), 4 models (NB, LR, SVM, BiLSTM), benchmarking, Streamlit deployment, unit testing

## 2. Methodology

### 2.1 Dataset
**Source:** UCI SMS Spam Collection | **Size:** 5,574 (Ham: 4,827/86.6%, Spam: 747/13.4%) | **Split:** 80/20 stratified

### 2.2 Preprocessing Pipeline
6-step process: (1) Lowercase (2) URL→"URL" (3) Email→"EMAIL" (4) Numbers→"NUM" (5) Remove punctuation (6) Normalize whitespace
**Example:** "URGENT! Call 12345 at www.site.com!!!" → "urgent call NUM at URL"

### 2.3 Feature Engineering
**TF-IDF:** max_features=5000, ngram_range=(1,2), captures phrases like "free prize", "call now"  
**BiLSTM Tokenization:** vocab=5000, maxlen=100, padding='post', oov_token='<OOV>'

### 2.4 Models
**Naive Bayes:** Probabilistic, fast baseline | **Logistic Regression:** Linear with L2 regularization | **Linear SVM:** Optimal hyperplane, high-dimensional | **BiLSTM:** Embedding(64d) → BiLSTM(64) → Dense(64) → Output(sigmoid), Adam optimizer, 5 epochs

### 2.5 Evaluation Metrics
Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix

## 3. Data Analysis

**Message Length:** Spam 1.94× longer (138.7 vs 71.5 chars avg) | **Class Imbalance:** 6.46:1 (Ham:Spam)

**Top Spam Words:** call, free, txt, claim, prize, urgent | **Top Ham Words:** to, i, you, thanks, love

**Spam Bigrams:** "call now", "free text", "txt stop" | **Ham Bigrams:** "let me", "see you", "thanks for"

## 4. Implementation

**Environment:** Python 3.13, VS Code | **Libraries:** scikit-learn (ML), TensorFlow (NN), pandas, numpy, matplotlib, seaborn, streamlit, pytest

**Code Structure:** Modular design with separate files for data loading, preprocessing, models, training, evaluation, web app

**Training Time:** NB: 0.02s | SVM: 0.05s | LR: 0.12s | BiLSTM: ~30s

## 5. Results

### 5.1 Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time |
|-------|----------|-----------|--------|----------|---------|------------|
| **Linear SVM** | **98.2%** | **98.9%** | **95.3%** | **97.1%** | **0.991** | 0.05s |
| Logistic Regression | 97.9% | 98.4% | 94.8% | 96.5% | 0.989 | 0.12s |
| BiLSTM | 97.7% | 98.1% | 94.5% | 96.3% | 0.987 | ~30s |
| Naive Bayes | 97.4% | 97.6% | 94.1% | 95.8% | 0.982 | 0.02s |

### 5.2 Linear SVM Confusion Matrix (Best Model)
```
              Ham    Spam
Actual  Ham   955     10    (98.9% correct)
        Spam    7    143    (95.3% correct)
```
**Implications:** Very low false positives (1.0%), good spam detection (95.3%), only 7 spam missed per 1,115 messages

### 5.3 Feature Importance (SVM)
**Top Spam:** free (+3.42), call (+3.18), txt (+2.97), claim (+2.84), prize (+2.71)  
**Top Ham:** love (-2.91), thanks (-2.43), sorry (-2.18), see you (-2.05), let me (-1.89)

### 5.4 Winner: Linear SVM
Best accuracy, fast training, low false positives, efficient prediction, suitable for production

## 6. Discussion

### 6.1 Key Findings
1. **Classical ML competitive** - SVM outperforms BiLSTM, much faster
2. **Preprocessing critical** - Text cleaning, normalization, bigrams essential
3. **TF-IDF effective** - Simple but powerful, computationally efficient
4. **Class imbalance manageable** - Despite 6.5:1 ratio, high precision maintained

### 6.2 Strengths
Multiple model comparison | Rigorous evaluation | Production code + tests | Web deployment | Complete documentation

### 6.3 Limitations
- Small dataset (5,574) | English-only | Temporal drift | Basic TF-IDF features | No online learning

### 6.4 vs. Literature
Our SVM: 98.2% | Typical: 95-98% | BERT SOTA: 98-99% → Competitive with simpler approach

### 6.5 Practical Implications
**Deployment:** Linear SVM recommended - edge deployment possible, mobile-ready, no GPU needed  
**Maintenance:** Periodic retraining, monitor FP/FN rates

### 6.6 Lessons
Start simple (classical often sufficient) | Comprehensive evaluation essential | Low false positives crucial for UX

## 7. Conclusions

**Summary:** Successfully developed 4 models achieving >97% accuracy. Linear SVM winner: 98.2% accuracy, 98.9% precision, 95.3% recall, 0.05s training.

**Achievements:** Multiple models | 6-step preprocessing | Multi-metric evaluation | Streamlit deployment | Unit tests | Complete EDA

**Contributions:** Academic (comparative study, complete pipeline), Practical (production system, web UI), Educational (learning resource)

**Future Work:**  
- Short-term: SMOTE, metadata features, ensembles, hyperparameter tuning
- Long-term: BERT/transformers, multilingual, online learning, cloud API, LIME/SHAP
- Research: Temporal patterns, adversarial robustness, few-shot learning

**Key Takeaway:** Simple approaches can be best - Linear SVM (1990s algorithm) outperforms modern neural networks while being faster, simpler, more efficient.

## 8. References

1. SMS Spam Collection Dataset, UCI ML Repository
2. Pedregosa et al. (2011), Scikit-learn, JMLR
3. Abadi et al. (2015), TensorFlow
4. Salton & Buckley (1988), TF-IDF, Information Processing & Management
5. Joachims (1998), SVM for text, ECML
6. Hochreiter & Schmidhuber (1997), LSTM, Neural Computation
7. Almeida et al. (2011), SMS Spam Filtering, DocEng
