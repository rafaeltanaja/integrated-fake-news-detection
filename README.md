# integrated-fake-news-detection

# Integrated Fake News Detection

## Introduction

This repository contains the implementation of an integrated fake news detection system. The project aims to leverage machine learning techniques, integrating FakeBERT Classification, Text Style Analysis, and Credibility Verification to classify news articles as either fake or real. The approach involves data preprocessing, feature extraction, and model training using advanced algorithms.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/rafaeltanaja/integrated-fake-news-detection.git
   cd integrated-fake-news-detection
   ```
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the fake news detection model, execute the following command:

```bash
python main.py
```

Ensure that the required datasets are available in the appropriate directories.

## Methodology

The project follows these key steps:

1. **Data Collection & Preprocessing**: Cleaning and tokenizing text data.
2. **Feature Engineering**: Extracting relevant features using NLP techniques.
3. **Model Training**: Training machine learning models like XGBoost and FakeBERT.
4. **Evaluation**: Assessing model performance using accuracy, precision, recall, and F1-score.

## Limitations

Some challenges faced in this project include:

- The model is trained only on English-language datasets.
- Dataset bias towards political news.
- Limitations in keyword extraction affecting related news retrieval.
- API constraints on data scraping.

For further improvements, future work could explore multi-language support and more diverse datasets.

