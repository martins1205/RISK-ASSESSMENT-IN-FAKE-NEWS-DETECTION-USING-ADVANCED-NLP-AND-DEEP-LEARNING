# RISK-ASSESSMENT-IN-FAKE-NEWS-DETECTION-USING-ADVANCED-NLP-AND-DEEP-LEARNING
A robust fake news detection system built with advanced deep learning models, including BERT Transformers, MiniLM, LSTM, and BLSTM. This project achieves 99.97% accuracy in identifying fake news articles and provides real-time analysis through a user-friendly web interface.

## Table of Contents
Overview
Features
Dataset
Model Architectures
Performance Metrics
Installation and Setup
Usage
System Requirements
File Structure
License
Contributors

## Overview
This project implements state-of-the-art deep learning models for detecting fake news. It leverages advanced architectures like BERT , MiniLM , LSTM , and BLSTM to classify news articles as either authentic or misleading. The system is designed for both researchers and end-users, offering a seamless experience through a Streamlit-based web interface.

## Key features include:

Real-time news article analysis
Advanced text preprocessing and feature extraction
Detailed credibility metrics and risk assessment
Support for English language
The best-performing model, BERT , achieves 99.97% accuracy , making it a reliable tool for combating misinformation.

# Features
Real-Time Analysis : Analyze news articles instantly with confidence scores.
Advanced Text Preprocessing : Clean and normalize input text using NLP techniques.
Deep Learning Models : Choose from multiple architectures (BERT, MiniLM, LSTM, BLSTM).
Detailed Metrics : View probability breakdowns, content credibility features, and improvement suggestions.
User-Friendly Interface : Built with Streamlit for an intuitive and interactive experience.
Risk Assessment : Evaluate false positive rates (FPR), false negative rates (FNR), and overall error rates (OER).
# Dataset
The project uses two datasets:

Fake.csv: Contains fake news articles.
True.csv: Contains legitimate news articles.
These datasets are preprocessed to ensure high-quality training and evaluation.

# Model Architectures
1. BERT
Base BERT model with a custom classification head.
Includes:
Dropout layer (0.1).
Two linear layers with ReLU activation.
Achieves 99.9% accuracy and the lowest risk scores:
FPR: 0.0
FNR: 0.0009
OER: 0.0005
2. MiniLM
Microsoft's lightweight MiniLM base model.
Optimized for efficiency and performance.
Achieves 99.9% accuracy .
3. LSTM
Embedding layer followed by multiple LSTM layers with dropout.
Dense layers for classification.
Achieves 99.4% accuracy .
4. BLSTM
Bidirectional LSTM architecture for context-aware processing.
Multiple layers with dropout.
Achieves 99.8% accuracy .
##  Performance Metrics
MODEL           ACCURACY              FALSE POSITIVE RATE (FPR)    FALSE NEGATIVE RATE (FNR)        OVERALL ERROR RATE (OER)
BERT            99.97%                  0.0                             0.0009                          0.0005
MiniLM          99.9%                   0.0006                          0.0005                          0.0005
LSTM            99.4%                   0.0101                          0.0019                          0.0056
BLSTM           99.8%                   0.0006                          0.0028                          0.0018

# Installation and Setup
1. Install Dependencies
Install the required packages using the following command:

bash
Copy
1.
Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. # Download the Model
Ensure the trained BERT model file and its associated files are in the project directory.

3. # Prepare the Data
Place the datasets Fake.csv and True.csv in the project directory and run the data preprocessing scripts.
4. # Project Structure
- `Fake.csv`: Dataset containing fake news articles
- `True.csv`: Dataset containing true news articles 
- `bestapp.py`: Best performing model implementation
- `requirements.txt`: List of Python packages required for the project
- `new project.ipynb`: Development notebook
# Usage
1. Run the Jupyter notebook:
   ```bash
   jupyter notebook "new project.ipynb"
   ```
Run the Streamlit app:
bash
Copy
1
streamlit run bestapp.py
Access the application at:
Copy
1
http://localhost:8501
Paste a news article into the text area and click "🔍 Analyze Authenticity" to get results.
System Requirements
Python : 3.8+
Hardware : CUDA-compatible GPU (optional, for faster processing)
RAM : Minimum 4GB
Disk Space : 500MB


##  File Structure
├── bestapp.py                  # Main application file
├── bert_best_model.safetensors # Trained BERT model weights
├── Fake.csv                    # Fake news dataset
├── True.csv                    # True news dataset
├── requirements.txt            # Package dependencies
└── README.md                   # Project documentation

## Dataset
Dataset: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## License
This project is licensed under the MIT License . See the LICENSE file for more details.

## Contributors
This project is maintained and developed by:
Martins Adegbaju
Researcher and Developer
Email: [gbajumartins@gmail.com]
