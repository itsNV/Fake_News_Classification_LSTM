
# Fake News Classification using Bi-LSTM (Title-Based Prediction)

## Project Overview
This project focuses on **Fake News Classification** using **Natural Language Processing (NLP)** and **Bidirectional - Long Short-Term Memory (Bi-LSTM)** networks.  
The goal is to predict whether a news article is **fake or real** by analyzing **only the news title**, without using the full article text.

This approach demonstrates how much contextual information can be extracted from short text sequences like headlines.

---

## Problem Statement
Given a news **title**, the task is to classify it as:
- **Fake News**
- **Real News**

This is a **binary text classification problem**, where the model learns linguistic patterns commonly found in misleading or fake headlines.

---

## Dataset Description
The dataset contains the following features:

| Feature Name | Description |
|--------------|-------------|
| `title`      | Headline of the news article |
| `text`       | Full news content |
| `label`      | Target variable (0 = Fake, 1 = Real) |

**Only the `title` feature was used** to train the model.

---

## Tech Stack & Libraries
- Python
- NumPy
- Pandas
- TensorFlow / Keras
- Scikit-learn
- NLTK

---

## Project Workflow
1. **Data Cleaning**
   - Lowercasing text
   - Removal of punctuation and special characters
   - Stopword removal
   - Lemmatization

2. **Text Preprocessing**
   - Tokenization of news titles
   - Sequence padding
   - Vocabulary creation

3. **Model Architecture**
   - Embedding Layer
   - Bi-LSTM Layer
   - Dense Layer with Sigmoid activation

4. **Model Training**
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam
   - Validation split used during training

5. **Model Evaluation**
   - Validation Accuracy
   - Performance analysis on unseen data

---

## Model Performance
- **Validation Accuracy:** **0.89**
- The model effectively captures misleading linguistic patterns present in fake news headlines.

---

## Key Insights
- News titles alone contain strong signals for fake news detection.
- Bi-LSTM performs well in understanding sequential and contextual word dependencies.
- Using short text inputs reduces computational cost while maintaining high accuracy.

---

## Future Improvements
- Combine **title + text** for improved performance
- Apply **pretrained word embeddings** (GloVe, FastText)
- Fine-tune **Transformer models (BERT)**

---

## Repository Structure

├── data/

│ └── fake_news_dataset.csv

├── notebooks/

│ └── Fake_News_Class_LSTM_RNN.ipynb

├── README.md

---

## Author
**Nisarg Patel**  
Aspiring Data Scientist | NLP & Machine Learning Enthusiast  

*This project is built for learning, experimentation, and showcasing NLP skills.*

---

If you find this project useful, feel free to star the repository!
