# RNN for IMDB Sentiment Analysis

## Overview

This project implements a Recurrent Neural Network (RNN) using PyTorch for sentiment classification of IMDB movie reviews.

The model predicts whether a movie review expresses a positive or negative sentiment.

## Dataset

The project uses the IMDB movie review dataset.

After removing duplicate records, the dataset contains **49,582 reviews**.

## Text Preprocessing

The notebook performs the following preprocessing steps:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove punctuation and HTML tags.
4. Remove stopwords.
5. Apply stemming using PorterStemmer.

## Feature Extraction

TF-IDF is used to convert the processed reviews into numerical features.

- Maximum features: 5,000
- Training samples: 39,665
- Testing samples: 9,917
- Train-test split: 80:20

## Model Architecture

The project uses a Vanilla RNN with PyTorch.

| Layer | Configuration |
|---|---|
| Input | 5,000 TF-IDF features |
| RNN | `nn.RNN`, hidden size 128, one layer |
| Fully connected | 128 → 1 |
| Output | Sigmoid for binary classification |
| Classification threshold | 0.5 |

**Implementation note:** In the current notebook, each TF-IDF feature vector is given to the RNN with a sequence length of 1. TF-IDF also does not preserve the original word order. Therefore, this implementation does not learn word-by-word sequence patterns in the usual way a text RNN does.

## Model Performance

Test accuracy: **85.37%**

## Technologies Used

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- NLTK

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/DeepakYadav9489/CNN-RNN.git
   ```

2. Install the required packages:

   ```bash
   pip install torch pandas numpy scikit-learn nltk
   ```

3. Open `RNN_For_SentimentAnalysis.ipynb` in Jupyter Notebook or Google Colab.

4. Run the notebook cells in order.

## Project File

- `RNN_For_SentimentAnalysis.ipynb` – notebook containing text preprocessing, TF-IDF feature extraction, RNN training, and evaluation.

## Future Improvements

- Tokenize reviews into word or subword sequences.
- Use an embedding layer instead of TF-IDF vectors.
- Pass sequences with their actual time steps to the RNN.
- Compare Vanilla RNN, LSTM, and GRU models.
