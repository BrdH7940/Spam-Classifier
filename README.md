# Email Spam Classification

**Lab 3 - Mathematics for AI - HCMUS**

## 📧 Project Overview

This project implements email spam classification using machine learning techniques. The goal is to automatically distinguish between spam and legitimate (ham) emails by analyzing their content, specifically the subject line and message body.

## 📊 Dataset

### Enron-Spam Dataset

We use the **Enron-Spam** dataset, which contains real emails from the Enron Corporation with manual spam/ham labels.

### Dataset Properties:

- **Training Set**: 27,284 emails
- **Validation Set**: 3,084 emails
- **Features**: Subject line and Message body
- **Target Classes**:
  - `spam` (13,858 emails in training set)
  - `ham` (13,426 emails in training set)
- **Class Distribution**: Nearly balanced (~50.8% spam, ~49.2% ham)

### Data Challenges:

- **Missing Values**:
  - Training set: 229 missing subjects, 352 missing messages
  - Validation set: 29 missing subjects, 35 missing messages
- **Text Preprocessing**: Combined subject and message into a single text field
- **Feature Extraction**: Used Bag of Words (BoW) with CountVectorizer and English stop words removal

## 🤖 Machine Learning Models

### 1. Multinomial Naive Bayes

- **Implementation**: Custom from scratch using probabilistic approach
- **Key Features**:
  - Laplace smoothing (α = 1.0) to handle unseen words
  - Efficient batch processing for large datasets
  - Calculates P(spam|text) using Bayes' theorem
  - Handles sparse matrices for memory efficiency

### 2. Logistic Regression

- **Implementation**: Custom from scratch using gradient descent
- **Key Features**:
  - Sigmoid activation function with numerical stability (clipping)
  - Mini-batch gradient descent (batch size = 1000)
  - Learning rate = 0.1, Maximum iterations = 1500
  - Cross-entropy loss function
  - Validation loss monitoring for overfitting detection

## 📈 Results

### Naive Bayes Performance:

| Dataset    | Accuracy   | Precision  | Recall     | F1-Score   |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| Training   | 99.27%     | 99.26%     | 99.30%     | 99.28%     |
| Validation | **99.03%** | **98.92%** | **99.17%** | **99.04%** |

### Logistic Regression Performance:

| Dataset    | Accuracy   | Precision  | Recall     | F1-Score   |
| ---------- | ---------- | ---------- | ---------- | ---------- |
| Training   | 99.66%     | 99.38%     | 99.96%     | 99.67%     |
| Validation | **98.87%** | **98.29%** | **99.49%** | **98.89%** |

### Key Observations:

- **Naive Bayes** achieved slightly better validation performance with 99.03% accuracy
- **Logistic Regression** showed excellent training convergence with 96.04% loss reduction
- Both models demonstrated high precision and recall, indicating robust spam detection
- Minimal overfitting observed in both models

## 🎓 What I Learned

### Technical Skills:

1. **Machine Learning from Scratch**: Implemented both Naive Bayes and Logistic Regression without using sklearn, gaining deep understanding of the mathematical foundations
2. **Numerical Optimization**: Learned gradient descent optimization, batch processing, and numerical stability techniques (sigmoid clipping, epsilon smoothing)

### Domain Insights:

1. **Spam Detection**: Understood the characteristics that distinguish spam from legitimate emails
2. **Feature Engineering**: Learned the importance of text preprocessing and feature selection in NLP
3. **Class Imbalance**: Dealt with near-balanced datasets and appropriate evaluation metrics
4. **Real-world Applications**: Gained appreciation for the practical challenges in email security systems

This project provided hands-on experience with the complete machine learning pipeline, from data preprocessing to model deployment, while reinforcing fundamental mathematical concepts in a practical application.
