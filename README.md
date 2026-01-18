# Comprehensive Arabic Sentiment Analysis: From Morphology to Transformers

**A full-scale NLP pipeline for sentiment classification on 300K Arabic reviews.**  
This project benchmarks the effectiveness of classical preprocessing techniques (morphological analysis, POS tagging, NER) against a range of models—from traditional classifiers (LR, KNN) to transformer-based BERT.

**Dataset Sources**:  
[Kaggle: 330k Arabic Sentiment Reviews](https://www.kaggle.com/datasets/abdallaellaithy/330k-arabic-sentiment-reviews)

---

## 🚀 Key Features & Scientific Contributions

*   **Large-Scale Analysis**: Processes and evaluates 300,000+ Arabic reviews across multiple domains.
*   **Preprocessing Benchmark**: Systematically measures the impact of advanced linguistic preprocessing (Morphological analysis, POS tagging, NER) on final model performance.
*   **Efficiency vs. Performance**: Demonstrates that **training on a subset (10k records) using CPU-efficient traditional models yields comparable results (~89% Accuracy/F1)** to full-dataset training with BERT on GPU.
*   **Modular Pipeline**: Code is structured into parallel experimental tracks, separating data preparation, training, and evaluation.

---

## 📂 Repository Structure & Workflow

The project follows a **Parallel Experimentation** workflow. Each track operates independently to isolate the effects of specific preprocessing techniques.

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
│   └── modeling/                 # Model training & evaluation (Model_*.ipynb)
├── README.md                     # Project documentation
└── LICENSE                       # MIT License
```

### ⚠️ Technical Note: Parallel CPU Processing & Memory Management

To achieve efficient processing without GPUs for the traditional tracks, we utilized **`pandarallel`**, which enables parallel processing on CPUs.

*   **Parallelism Strategy**:
    *   We run independent parallel workers (cores).
    *   **Function Self-Containment**: You will notice that libraries (e.g., `stanza`, `ISRIStemmer`) and regex patterns are **re-imported/re-defined inside the processing functions**. This is intentional. It ensures that each child process has its own independent instance of the dependencies and data, avoiding race conditions and serialization issues common in Python's multiprocessing.
    *   **Memory (RAM) Constraints**: Some notebooks explicitly reduce the number of workers (e.g., `n_workers = 3`) instead of using all available cores. This is a strategic decision to prevent RAM saturation when loading large language models (like Stanza) for each worker simultaneously.

### 📘 Educational Structure
Each preprocessing notebook (`*_preprocessing.ipynb`) is structured as a guided tutorial:
*   **Step-by-Step Cells**: Every processing stage (Cleaning, Tokenization, POS Tagging) is isolated in its own cell.
*   **Statistical Foundation**: metrics and timing are printed for each step to justify the approach and show the computational cost of each linguistic feature.
*   **Beginner Friendly**: The code demonstrates how to set up parallel NLP pipelines from scratch.

---

## 🧪 Methodology & Pipelines

### 1. Dataset
*   **Size**: Total 330,000 reviews; Sub-sampled to 10,000 for efficient CPU experimentation in traditional tracks.
*   **Labels**: Binary sentiment (Positive/Negative).

### 2. Experimental Tracks

#### A. Sequential Track (Baseline)
*   **Preprocessing**: `Sequential Processing.ipynb`
    *   Basic cleaning, removal of diacritics, English content filtering.
*   **Modeling**: `Sequential_Model.ipynb`
    *   Trained Logistic Regression (LR), KNN, SVM.
    *   **Result**: Establishes the baseline performance.

#### B. Gensim (Embeddings) Track
*   **Preprocessing**: `Gensim_PreProcessing.ipynb`
    *   Trains custom **Word2Vec** and **FastText** models on the corpus.
    *   Generates averaged word embeddings content.
*   **Modeling**: `Gensim_model.ipynb`
    *   Uses embedding vectors as features for classification.

#### C. POS (Part-of-Speech) Track
*   **Preprocessing**: `POS_PreProcessing.ipynb`
    *   Enriches text with **Stanza** POS tags.
    *   Combines text stems with grammatical features.
*   **Modeling**: `POS_Model.ipynb`
    *   Evaluates if grammatical structure correlates with sentiment.

#### D. POS + NER (Named Entity Recognition) Track
*   **Preprocessing**: `POS_NER_Preprocessing.ipynb`
    *   Extracts Named Entities (Person, Location, Organization) alongside POS tags.
    *   Hypothesis: Entities might carry strong sentiment signals.
*   **Modeling**: `POS_NER_Model.ipynb`
    *   Uses rich linguistic features for training.

#### E. BERT Track (State-of-the-Art)
*   **Modeling**: `Bert_Model.ipynb`
    *   Fine-tunes `aubmindlab/bert-base-arabertv02` on the dataset using GPU.
    *   Generates deep contextual embeddings.

#### F. BERT Partial Fine-Tuning (Final Best Model)
*   **Modeling**: `Bert_Model_Partial_FineTune.ipynb`
    *   **Approach**: Partial fine-tuning of `aubmindlab/bert-base-arabertv02` (last 2 layers + pooler) followed by Bi-Directional LSTM/RNN heads.
    *   **Outcome**: Using the **full dataset**, this approach achieved the **best accuracy** and performance among all experiments, serving as the final output of this project.

---

## 📊 Results & Findings

We compared the performance of traditional Machine Learning models (trained on 10k samples via CPU) against the heavy-weight BERT model (trained on GPU).

| Track | Model | Accuracy | F1-Score | Training Resource |
| :--- | :--- | :--- | :--- | :--- |
| **Sequential** | SVM / LR | **~89%** | **~89%** | CPU (Fast) |
| **Gensim** | LR (FastText) | ~89% | ~89% | CPU |
| **POS Tagging** | Logistic Regression | ~89% | ~89% | CPU |
| **POS + NER** | Logistic Regression | ~89% | ~89% | CPU |
| **BERT** | Arabert v02 | ~90% | ~90% | GPU (Heavy) |
| **BERT (Partial FT)** | **Bi-LSTM Head** | **Highest (Best)** | **Highest (Best)** | GPU (Optimized) |

### Key Insights
1.  **Best Performance**: The **BERT Partial Fine-Tuning** model with a Bi-LSTM head achieved the highest accuracy on the full dataset, outperforming all other approaches.
2.  **Efficiency Wins**: Traditional Morphological analysis and simple classifiers (Logistic Regression, SVM) achieved **~89% accuracy**, matching the performance of standard Transformer implementations.
2.  **Resource Usage**: We achieved these results training on only **10,000 records** on a standard CPU, proving that massive compute isn't always necessary for high-quality sentiment analysis.
3.  **Feature Engineering**: The consistency of the results across tracks (Sequential vs. POS/NER) suggests that the core sentiment signal in Arabic reviews is robust and easily captured by diverse feature engineering approaches.

---

## 🛠️ How to Reproduce

### Prerequisites
*   Python 3.8+
*   `pandas`, `scikit-learn`, `gensim`, `stanza`, `torch`, `transformers`, `pandarallel`

### Steps
1.  **Download Data**: Download the dataset from Kaggle and place `arabic_sentiment_reviews.csv` in `data/raw/`.
2.  **Run Preprocessing**: Execute notebooks in `notebooks/preprocessing/` to generate the processed CSVs in `data/processed/`.
3.  **Run Modeling**: Execute notebooks in `notebooks/modeling/` to train and evaluate models.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
