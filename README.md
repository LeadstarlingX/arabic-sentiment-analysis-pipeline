# Arabic Sentiment Analysis NLP Project

This project explores various Natural Language Processing (NLP) techniques to classify Arabic sentiment reviews. It implements multiple preprocessing and modeling tracks, comparing Sequential, Parallel, Gensim (Word2Vec/FastText), POS-tagging, and BERT-based approaches.

## 📂 Project Structure

The project is organized into three main directories to separate code, data, and artifacts.

```text
NLP_Project/
├── data/                         # Data directory (Ignored by Git)
│   ├── raw/                      # Place 'arabic_sentiment_reviews.csv' here
│   └── processed/                # Generated CSVs from preprocessing notebooks
├── models/                       # Model directory (Ignored by Git)
│   ├── arabic_w2v.model          # Word2Vec model
│   ├── arabic_ft.model           # FastText model
│   ├── best_model.pkl            # Best ML model (SVM/LR/etc.)
│   └── *.pt                      # BERT embeddings
├── notebooks/                    # Jupyter Notebooks
│   ├── preprocessing/            # Data cleaning & feature extraction
│   └── modeling/                 # Model training & evaluation
├── README.md                     # Project documentation
└── .gitignore                    # Git configuration
```

## 🚀 Workflow & Pipelines

This project uses a **Parallel Experimentation** workflow. Each track operates independently, typically starting from the raw dataset.

### 1. Sequential Track
*   **Goal**: Baseline preprocessing and modeling.
*   **Notebooks**:
    *   `preprocessing/Sequential_Processing.ipynb`: Cleans raw text, filters English content.
    *   `modeling/Sequential_Model.ipynb`: Trains simple classifiers (KNN, SVM, LR) on TF-IDF features.

### 2. Gensim (Embeddings) Track
*   **Goal**: Utilize word embeddings (Word2Vec, FastText) for feature representation.
*   **Notebooks**:
    *   `preprocessing/Gensim_PreProcessing.ipynb`: Trains custom W2V/FastText models on the corpus.
    *   `modeling/Gensim_model.ipynb`: Uses averaged embeddings for classification.

### 3. Part-of-Speech (POS) Track
*   **Goal**: Enhance text features with grammatical tags using Stanza.
*   **Notebooks**:
    *   `preprocessing/POS_PreProcessing.ipynb`: Extracts POS tags.
    *   `modeling/POS_Model.ipynb`: Uses POS tags + N-grams for classification.

### 4. POS + NER (Named Entity Recognition) Track
*   **Goal**: Further enhancement with Entity Recognition.
*   **Notebooks**:
    *   `preprocessing/POS_NER_Preprocessing.ipynb`: Extracts NER tags alongside POS.
    *   `modeling/POS_NER_Model.ipynb`: Trains models using rich linguistic features.

### 5. BERT Track
*   **Goal**: State-of-the-art Deep Learning approach.
*   **Notebooks**:
    *   `modeling/Bert_Model.ipynb`: Fine-tunes `aubmindlab/bert-base-arabertv02` for sentiment analysis.

## 🛠️ Setup & Usage

1.  **Install Dependencies**:
    Ensure you have `pandas`, `sklearn`, `gensim`, `stanza`, `torch`, `transformers`, and `pandarallel` installed.
    ```bash
    pip install pandas scikit-learn gensim stanza torch transformers pandarallel
    ```

2.  **Download Dataset**:
    *   Place your `arabic_sentiment_reviews.csv` file into the `data/raw/` directory.

3.  **Run Preprocessing**:
    *   Open any bucket in `notebooks/preprocessing/` and run all cells.
    *   This will generate processed CSVs in `data/processed/`.

4.  **Run Modeling**:
    *   Open the corresponding notebook in `notebooks/modeling/`.
    *   **Note**: Ensure the input path in `pd.read_csv()` points to the correct processed file in `data/processed/`.

## ⚠️ Notes
*   **Large Files**: The `data/` and `models/` directories are excluded from version control via `.gitignore` to avoid repository bloat.
*   **Stanza**: The Stanza pipelines usually require downloading language models on the first run.
