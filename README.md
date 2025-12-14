# Tunisian Dialect Classification

A machine learning project for classifying Tunisian dialect tweets using transformer-based models and MARBERT tokenization.

## 📋 Project Overview

This project analyzes and classifies Tunisian dialect text from social media (tweets) using state-of-the-art Arabic NLP models. The project includes comprehensive text preprocessing, tokenization analysis, and prepares the foundation for dialect classification tasks.

## 🚀 Features

- **Dataset Processing**: Automated loading and processing of the Tunisian Dialect Corpus
- **MARBERT Tokenization**: Advanced Arabic text tokenization using UBC-NLP's MARBERT model
- **Statistical Analysis**: Token length distribution analysis with percentile calculations
- **Data Export**: CSV export functionality for further analysis and model training
- **Tweet Classification**: Binary classification setup for Tunisian dialect detection

## 📊 Dataset

- **Source**: [arbml/Tunisian_Dialect_Corpus](https://huggingface.co/datasets/arbml/Tunisian_Dialect_Corpus)
- **Content**: Tunisian dialect tweets with binary labels
- **Columns**: 
  - `Tweet`: Raw tweet text in Tunisian dialect
  - `label`: Classification labels for dialect detection
- **Language**: Tunisian Arabic dialect

## 🛠️ Technologies Used

- **Python 3.x**
- **Transformers**: HuggingFace transformers library
- **MARBERT**: Multi-dialectal Arabic BERT model (UBC-NLP/MARBERT)
- **Datasets**: HuggingFace datasets library
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and statistical analysis

## 📋 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install transformers datasets pandas numpy torch
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Tunisian-dialect-classification.git
   cd Tunisian-dialect-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**:
   ```bash
   jupyter notebook main.ipynb
   ```
   Or open in your preferred notebook environment.

## 📁 Project Structure

```
├── main.ipynb                    # Main analysis and preprocessing notebook
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
├── tunisian_dialect_corpus.csv   # Exported dataset (generated after running)
└── .gitignore                    # Git ignore file
```

## 🔍 Analysis Pipeline

1. **Data Loading**: Load the Tunisian Dialect Corpus from HuggingFace
2. **Data Exploration**: Examine tweet content and label distributions
3. **Tokenization**: Process tweets using MARBERT tokenizer
4. **Statistical Analysis**: Calculate token length percentiles (50th, 75th, 90th, 95th, 99th)
5. **Data Export**: Save processed data to CSV for model training

## 📈 Token Length Analysis

The project includes comprehensive analysis of tweet token lengths:
- Median token length (50th percentile)
- Upper quartile (75th percentile) 
- 90th, 95th, and 99th percentile analysis
- Helps determine optimal sequence lengths for model training

## 🎯 Use Cases

- **Dialect Detection**: Identify Tunisian dialect in Arabic text
- **Social Media Analysis**: Analyze dialectal variations in tweets
- **NLP Research**: Foundation for Arabic dialect classification models
- **Data Preprocessing**: Clean and tokenize Arabic social media text

## 🔬 Next Steps

- Implement classification models (BERT, RoBERTa, etc.)
- Add data visualization and exploratory analysis
- Implement evaluation metrics and model comparison
- Add cross-validation and hyperparameter tuning

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports and feature requests
- Improve documentation
- Add new analysis features
- Optimize preprocessing pipeline

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [ArbML](https://github.com/ARBML) for providing the Tunisian Dialect Corpus
- [UBC-NLP](https://github.com/UBC-NLP) for the MARBERT model
- HuggingFace for the transformers and datasets ecosystem
- The Arabic NLP community for advancing dialectal Arabic processing

## 📚 References

- MARBERT: [Multi-dialectal Arabic BERT](https://arxiv.org/abs/2101.01785)
- Tunisian Dialect Corpus: [Dataset Paper](https://aclanthology.org/2020.wanlp-1.4/)

---

**Note**: This project is designed for research and educational purposes in Arabic NLP and computational linguistics.

