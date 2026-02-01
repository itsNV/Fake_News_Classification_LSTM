
# Fake News Classification using LSTM (Text Pair Classification)

## Project Overview
This project focuses on **Fake News Classification / Text Pair Classification** using **Recurrent Neural Networks (LSTM)**.  
The objective is to analyze two textual descriptions and determine whether they refer to the **same security / same news context** or not.

The project leverages deep learning–based NLP techniques to capture semantic relationships between paired text inputs.

---

## Problem Statement
Given two textual inputs:
- `description_x`
- `description_y`

The task is to predict whether both descriptions convey the **same information (same security / same news)** or represent **different or misleading (fake) content**.

This is treated as a **binary classification problem**.

---

## Dataset Description
The dataset contains the following features:

| Feature Name     | Description |
|------------------|-------------|
| `description_x`  | First news/security description (raw text) |
| `description_y`  | Second news/security description (raw text) |
| `token_x`        | Tokenized version of `description_x` |
| `token_y`        | Tokenized version of `description_y` |
| `same_security`  | Target variable (1 = Same, 0 = Different/Fake) |

For model training, **only `description_x` and `description_y`** were used as input features.

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
   - Lowercasing
   - Removal of special characters
   - Stopword removal
   - Lemmatization

2. **Text Preprocessing**
   - Tokenization
   - Padding sequences
   - Vocabulary creation

3. **Feature Engineering**
   - Combined semantic representation of `description_x` and `description_y`

4. **Model Architecture**
   - Embedding Layer
   - LSTM Layer
   - Fully Connected (Dense) Layers
   - Sigmoid activation for binary classification

5. **Model Training**
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy
   - Validation split applied

6. **Evaluation**
   - Validation Accuracy
   - Classification Metrics

---

## Model Performance
- **Validation Accuracy:** **0.89**
- Model effectively captures semantic similarity between paired news descriptions.

---

## Results & Insights
- LSTM performs well for contextual understanding of long text sequences.
- Using paired descriptions improves fake news and similarity detection.
- Deep learning outperforms traditional ML models for this task.

---

## Future Improvements
- Use **Bi-LSTM** or **GRU**
- Experiment with **pretrained embeddings** (GloVe, FastText)
- Apply **Transformer-based models (BERT)**
- Handle class imbalance using advanced sampling techniques

---

## Repository Structure

├── data/

  └── train.csv
  
  └── test.csv

├── notebooks/

  └── Fake_News_Class_LSTM_RNN.ipynb

├── README.md

  └── requirements.txt



---

## Author
**Nisarg Patel**  
Aspiring Data Scientist | NLP & Machine Learning Enthusiast  

*This project is built for learning, experimentation, and showcasing NLP skills.*

---

If you find this project useful, feel free to star the repository!
