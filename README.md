# 🏦 P2P Lending Default Prediction using Deep Learning

## 📌 Project Overview
This project aims to predict credit default in peer-to-peer lending using deep learning models that analyze both structured borrower data and unstructured text (loan descriptions). This is a Final Year Project (FYP) for Bachelor of Computer Science.

## 🎯 Objectives
1. Reproduce results from the baseline paper: "Credit default prediction from user-generated text in peer-to-peer lending using deep learning" (Kriebel & Stitz, 2022)
2. Improve upon the baseline deep learning models
3. Develop an AI-powered mobile application for credit risk assessment

## 🧠 Deep Learning Models
We will implement and compare 6 different architectures:
1. **Average Embedding NN** - Baseline model
2. **CNN** (Convolutional Neural Network) - Pattern recognition in text
3. **RNN** (Recurrent Neural Network) - Sequential text analysis
4. **CNN + RNN** - Hybrid approach
5. **BERT** - Transformer-based model
6. **RoBERTa** - Optimized BERT variant

## 📊 Dataset
- **Source**: Lending Club historical loan data
- **Features**: Structured data (income, credit score, loan amount) + Unstructured text (borrower descriptions)
- **Target**: Binary classification (Default / No Default)

## 🗂️ Project Structure
```
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks for exploration and modeling
├── src/                   # Source code
│   ├── preprocessing/     # Data cleaning and preparation
│   ├── models/            # Deep learning model implementations
│   └── evaluation/        # Model evaluation scripts
├── app/                   # Mobile application code
├── docs/                  # Documentation and reports
└── results/               # Model results and visualizations
```

## 📅 Timeline
- **Week 1-2**: Literature review and data exploration
- **Week 3-6**: Model implementation and training
- **Week 7-9**: Model improvement and optimization
- **Week 10-12**: Mobile app development
- **Week 13-14**: Final testing and documentation

## 👥 Team
-  Team Lead
- Team Member 2
- Team Member 3

## 📚 References
Kriebel, J., & Stitz, L. (2022). Credit default prediction from user-generated text in peer-to-peer lending using deep learning. European Journal of Operational Research. https://doi.org/10.1016/j.ejor.2021.12.024

---
**Status**: 🚀 In Progress | **Last Updated**: December 2025
