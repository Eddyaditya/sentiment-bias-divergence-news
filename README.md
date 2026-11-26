# Where Sentiment and Bias Diverge

Code repository for the study:

**Where Sentiment and Bias Diverge: A Multidimensional Computational Analysis of Indian News Coverage of a Protest Movement**

Authors: **Aditya Negi and Ashwini Kumar Singh**

## Overview

This repository contains the analysis and figure-generation code used to study news coverage of the CJP protest movement across nine Indian and international news outlets.

The final corpus contains **3,857 articles published between May and September 2026**. Six Indian outlets form the primary comparison set, while BBC, The Guardian, and CNN are retained as descriptive international comparators.

The analysis examines multiple dimensions separately rather than treating any single measure as a definitive measure of media bias:

- Article and headline sentiment
- Named-entity salience
- Entity-proximate sentiment
- Article-versus-entity sentiment divergence
- Distinctive vocabulary using TF-IDF
- Sentence-level media-bias signals using MAGPIE/BABE
- Cross-method robustness analysis

## Dataset

The dataset used in the study is publicly available on Kaggle:

https://www.kaggle.com/datasets/adityanegi517/cjp-protest-news-coverage-dataset-2026

The corpus contains article metadata, text, source information, publication dates, URLs, and corpus-selection information.

Copyright in the original news content remains with the respective publishers and authors.

## Analysis Pipeline

The scripts are organised in approximate execution order.

### Core Analysis

**01_sentiment_inference_kaggle.py**  
Runs article-body and headline sentiment inference using a RoBERTa-Large three-class sentiment model.

**02_sentiment_summary.py**  
Produces article-level sentiment summaries and outlet-level sentiment statistics.

**03_ner_entity_salience_kaggle.py**  
Performs named-entity recognition and generates entity-salience information using spaCy.

**04_entity_canonicalisation.py**  
Applies conservative canonicalisation to manually specified unambiguous entity-name variants.

**05_entity_proximate_sentiment_kaggle.py**  
Scores sentences containing retained person-name variants using the same sentiment model.

**06_article_vs_entity_sentiment.py**  
Compares whole-article sentiment with entity-proximate sentiment and calculates relative negativity divergence.

**07_tfidf_lexical_profiling.py**  
Computes outlet-specific distinctive vocabulary using unigram and bigram TF-IDF features.

**08_magpie_babe_inference_kaggle.py**  
Runs the sentence-level MAGPIE model fine-tuned on BABE to obtain model-based biased-language signals.

**09_consolidate_results.py**  
Consolidates analysis outputs into the final result tables.

**10_framework_vs_magpie.py**  
Compares the editorial indicators with the MAGPIE/BABE signal and produces cross-method association and robustness results.

### Publication Figures

**11_monthly_coverage_figure.py**  
Generates the monthly corpus distribution figure.

**12_sentiment_distribution_figure.py**  
Generates the outlet-level article sentiment distribution figure.

**13_article_vs_entity_figure.py**  
Generates the whole-article versus entity-proximate sentiment figure.

**14_magpie_wilson_figure.py**  
Generates the MAGPIE/BABE biased-sentence-share figure with Wilson intervals.

**15_negativity_vs_magpie_figure.py**  
Generates the negativity-versus-classifier scatter plot used for the robustness analysis.

## Models

### Sentiment

`j-hartmann/sentiment-roberta-large-english-3-classes`

Article bodies are processed in 400-token windows with a 50-token overlap. Window probabilities are averaged to obtain article-level sentiment probabilities.

### Named-Entity Recognition

spaCy transformer pipeline:

`en_core_web_trf`

### Sentence-Level Bias Signal

`mediabiasgroup/magpie-babe-ft`

The classifier output is treated as a **model-based signal of biased wording**, not as ground-truth evidence that an outlet is biased.

## Installation

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Install the spaCy transformer model:

```bash
python -m spacy download en_core_web_trf
```

GPU acceleration is recommended for the transformer-based sentiment, NER, entity-proximate sentiment, and MAGPIE/BABE inference stages.

## Reproducibility

Some model-inference scripts were originally executed in Kaggle GPU environments. File paths may therefore need to be adjusted when running the pipeline in a different environment.

The Kaggle dataset provides the corpus used in the study, while this repository provides the principal analysis and publication-figure code.

## Interpretation

The measures represent different constructs. Sentiment measures textual polarity, entity salience measures visibility, TF-IDF measures lexical emphasis, and MAGPIE/BABE provides model-generated sentence-level bias labels.

The study does **not** combine these measures into a single outlet-bias score and does not classify any news outlet as biased or unbiased.

## Requirements

Principal Python packages:

- pandas
- numpy
- matplotlib
- PyTorch
- Transformers
- spaCy
- scikit-learn

See `requirements.txt` for installation.

## Data and Copyright

The dataset is provided for academic research and reproducibility. Copyright in the original news articles remains with the respective publishers and authors. Inclusion of article text in the research corpus does not transfer ownership of the underlying content.

Users are responsible for complying with the terms and copyright requirements of the original publishers when using or redistributing article content.

## Contact

**Aditya Negi**  
Department of Computer Science and Engineering  
Graphic Era (Deemed to be University), Dehradun, India