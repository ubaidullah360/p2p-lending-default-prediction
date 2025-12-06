# Methodology

## 1. Introduction

This project replicates and extends the methodology presented in the baseline study by Kriebel and Stitz [1], which investigates credit default prediction using deep learning models applied to user-generated text from peer-to-peer lending platforms. Our implementation follows their approach systematically to ensure reproducibility and enable direct performance comparison.

## 2. Dataset

### 2.1 Data Source
We utilize the Lending Club dataset obtained from Kaggle [2], which contains historical loan data from one of the largest peer-to-peer lending platforms in the United States. The raw dataset comprises 2,260,701 loan records with 151 features spanning the period from 2007 to 2018 (Q4).

### 2.2 Dataset Characteristics
The Lending Club platform's business model connects borrowers with lenders through an online marketplace. During the loan application process, borrowers provide:
- Structured information (income, employment, home ownership, credit history)
- Unstructured text descriptions explaining their loan purpose and personal circumstances

Lending Club calculates internal credit scores and assigns interest rates based on application information and external credit bureau data. Lenders review this information to make funding decisions, typically diversifying risk across multiple borrowers [1].

## 3. Data Preprocessing

### 3.1 Temporal Scope Alignment

**Objective:** Restrict the dataset to match the exact time period used in the baseline study.

**Rationale:** 
The baseline study utilized data from 2007 to 2014 [1]. This period was specifically chosen because:
1. Loan descriptions became available on the Lending Club platform in 2007
2. Lending Club removed the option to provide loan descriptions in 2014 due to privacy concerns
3. Temporal alignment controls for macroeconomic variations (including the 2008 financial crisis)
4. Exact replication requires identical sample composition for valid performance comparison

**Implementation:**
- Initial dataset examination revealed 2,260,701 observations spanning 2007-2018
- The `issue_d` variable (loan issue date) was stored in "MMM-YYYY" format as object type
- 33 observations (0.0015% of total) contained missing issue dates
- Date conversion to datetime format enabled temporal filtering
- Year extraction facilitated period-based selection
- Applied filter: loans issued between January 2007 and December 2014 (inclusive)

**Outcome:** Temporal filtering significantly reduced the dataset size, retaining only loans issued during the 2007-2014 period for subsequent preprocessing steps.

### 3.2 Loan Status Filtering

**Objective:** Retain only loans with definitive outcomes to enable supervised learning.

**Rationale:**
Following the baseline methodology [1], we consider only loans that have reached completion, specifically:
- **Fully Paid:** Borrower successfully repaid the entire loan
- **Charged Off:** Loan is at least 120 days past due with no expectation of repayment (or earlier charge-off due to bankruptcy)

**Exclusions:**
Loans with indeterminate outcomes are excluded:
- Current (still being repaid)
- Late (16-30 days, 31-120 days)
- In Grace Period
- Default (separate minor category)
- Policy violations (do not meet credit policy requirements)

**Target Variable Definition:**
The binary target variable is defined as:
- **0 (Non-Default):** Fully Paid loans
- **1 (Default):** Charged Off loans

According to the baseline study, the final sample contains 14.5% charged-off loans and 85.5% fully paid loans [1], indicating a class imbalance that must be considered during model training.

### 3.3 Text Data Availability Filter

**Objective:** Retain only loan applications containing user-generated text descriptions.

**Rationale:**
The core innovation of this research lies in utilizing unstructured text data for credit default prediction. The baseline study reported that 125,798 loans in their database contained loan descriptions [1]. Loans without text descriptions cannot contribute to text-based model training and must be excluded.

**Implementation:**
- Filter dataset to retain only observations where the `desc` column (loan description) is non-null
- Verify text data quality and completeness

### 3.4 Text Length Filter

**Objective:** Ensure loan descriptions contain substantial information.

**Rationale:**
Following the baseline methodology [1], we restrict the sample to loan applications with descriptions containing at least 40 words. This threshold ensures that:
1. Descriptions provide meaningful information beyond trivial statements
2. Sufficient textual data exists for deep learning models to extract patterns
3. Sample composition matches the baseline study for valid comparison

**Note:** The baseline study noted that results are robust against differences in minimum description lengths [1].

**Implementation:**
- Tokenize text descriptions to count words
- Apply filter: retain only descriptions with word count ≥ 40
- Document the number of observations removed at this stage

### 3.5 Expected Sample Size

After applying all preprocessing filters (temporal scope, loan status, text availability, text length), the baseline study achieved a final sample size of 40,229 funded loans [1]. Our preprocessing aims to achieve a comparable sample size, with minor variations acceptable due to potential differences in data access timing or platform updates.

## 4. Feature Engineering

### 4.1 Feature Selection

Following the comprehensive variable set utilized in the baseline study [1], we extract 22 structured features plus user-generated text descriptions.

#### 4.1.1 Numerical and Binary Features (19 features)

| Feature | Dataset Column | Transformation Required |
|---------|---------------|------------------------|
| Adverse public records | `pub_rec` | None |
| Annual income | `annual_inc` | Winsorize at 0.01 and 0.99 |
| Debt-to-income ratio | `dti` | None |
| Delinquencies (2 years) | `delinq_2yrs` | None |
| Employment length unknown | `emp_length` | Create binary indicator |
| Employment title unknown | `emp_title` | Create binary indicator |
| FICO score | `fico_range_low`, `fico_range_high` | Calculate interval center |
| Income verified | `verification_status` | Create binary indicator |
| Inquiries within 6 months | `inq_last_6mths` | None |
| Installment to total income | `installment`, `annual_inc` | Calculate ratio |
| Loan amount | `loan_amnt` | None |
| Loan term | `term` | Extract numeric value |
| Length of credit history | `earliest_cr_line` | Calculate months from issue date |
| Revolving balance | `revol_bal` | None |
| Revolving line utilization rate | `revol_util` | None |
| Total open accounts | `open_acc` | None |
| Loan status (target) | `loan_status` | Convert to binary (0/1) |

**Note on Missing Features:**
The baseline study included two additional external macroeconomic features [1]:
- House Price Index (year-to-year change)
- Unemployment Rate (state-wide rate)

These features are not present in the Kaggle dataset and require merging with external economic databases. Following supervisor guidance, we contacted the original authors for data sourcing information. In the absence of response, we proceed with the 20 available features (90.9% of the original feature set). The core predictive power derives from borrower-specific characteristics and text data rather than macroeconomic context variables.

#### 4.1.2 Categorical Features (3 features)

| Feature | Dataset Column | Encoding |
|---------|---------------|----------|
| Home ownership | `home_ownership` | One-hot encoding |
| Loan purpose | `purpose` | One-hot encoding |
| Rating sub-grade | `sub_grade` | One-hot encoding |

#### 4.1.3 Text Feature

| Feature | Dataset Column | Preprocessing |
|---------|---------------|---------------|
| User-generated loan description | `desc` | See Section 4.2 |

### 4.2 Text Preprocessing Pipeline

Following the exact text preprocessing steps specified in the baseline study [1]:

**Step 1: Data Cleaning**
- Remove automatically generated logs and system-generated content
- Remove punctuation marks (except for subsequent punctuation proportion calculation)

**Step 2: Numeric Normalization**
- Replace all numbers with their corresponding word representations
- Replace dollar signs ($) with the word "dollar"

**Step 3: Case Normalization**
- Convert all letters to lowercase
- Exception: Skip this step for RoBERTa model (pre-trained on cased text)

**Step 4: Lemmatization**
- Map words to their inflected base forms
- Example: "running" → "run", "better" → "good"

**Step 5: Length Standardization**
- Limit descriptions to 115 words (95th percentile of description lengths) [1]
- Rationale: Retains most information while maintaining constant input dimensions
- Apply zero-padding for descriptions shorter than 115 words
- Zero-padding adds zero vectors to the end without altering content

**Step 6: Word Embeddings**
- Transform words into semantic vector representations
- Deep learning models learn domain-specific embeddings during training
- Embeddings map semantically similar words to similar vector positions

## 5. Train-Validation-Test Split

Following the baseline methodology [1], the dataset is divided into three distinct subsamples:

| Subset | Size | Purpose |
|--------|------|---------|
| Training Set | 20,000 observations | Model parameter training |
| Validation Set | 10,000 observations | Hyperparameter optimization and model selection |
| Test Set | 10,229 observations | Final out-of-sample performance evaluation |
| **Total** | **40,229 observations** | - |

**Split Method:** Random sampling (not time-based)

**Rationale:**
- Training set: Sufficient size for deep learning model convergence
- Validation set: Used to evaluate different hyperparameter configurations and select optimal models
- Test set: Provides unbiased estimate of final model performance on unseen data
- Three-way split prevents overfitting and ensures robust performance estimation

## 6. Deep Learning Architectures

This project implements and compares six deep learning architectures as specified in the baseline study [1]:

### 6.1 Convolutional Neural Network (CNN)
- **Architecture:** Embedding layer → Convolutional layers → Pooling layers → Dense layers
- **Strength:** Identifies local patterns and features in text regardless of position
- **Use Case:** Effective when relevant information is distributed across different text segments

### 6.2 Recurrent Neural Network (RNN)
- **Architecture:** Embedding layer → GRU (Gated Recurrent Unit) layers → Dense layer
- **Strength:** Processes text sequentially, maintaining memory of important passages
- **Use Case:** Captures word order and sequential dependencies

### 6.3 Convolutional Recurrent Neural Network (CNN+RNN)
- **Architecture:** Embedding layer → Convolutional layers → GRU layers → Dense layer
- **Strength:** Combines pattern recognition with sequential processing
- **Use Case:** Mimics human text processing by grouping words into concepts before sequential analysis

### 6.4 Average Embedding Neural Network
- **Architecture:** Embedding layer → Averaging layer → Dense layers
- **Strength:** Reduces dimensionality by averaging word embeddings
- **Use Case:** Simpler model requiring less training data

### 6.5 BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture:** Pre-trained BERT base uncased → Dropout layer → Dense layer
- **Strength:** Leverages transfer learning with attention mechanisms
- **Use Case:** State-of-the-art performance through pre-training on massive text corpora

### 6.6 RoBERTa (Robustly Optimized BERT)
- **Architecture:** Pre-trained RoBERTa base → Dropout layer → Dense layer
- **Strength:** Optimized BERT variant with enhanced pre-training
- **Use Case:** Improved performance over BERT on benchmark tasks

## 7. Evaluation Metrics

### 7.1 Primary Metric: Area Under the ROC Curve (AUC)

Following the baseline study [1], we evaluate model performance using the Area Under the Receiver Operating Characteristic (ROC) Curve.

**ROC Curve:**
- Plots True Positive Rate against False Positive Rate
- Evaluates classifier performance across all decision thresholds
- A strong classifier maintains high true positive rate while minimizing false positives

**AUC Interpretation:**
- Range: 0.5 (random prediction) to 1.0 (perfect prediction)
- Values above 0.7 indicate good predictive performance
- Baseline study achieved AUC values ranging from 0.612 (text only) to 0.712 (combined features) [1]

**Advantages:**
- Threshold-independent metric
- Robust to class imbalance
- Industry-standard for credit risk modeling

### 7.2 Statistical Testing

Performance comparisons between models utilize:
- DeLong test for AUC comparison
- Bonferroni correction for multiple testing
- Significance levels: *p* < 0.1, **p* < 0.05, ***p* < 0.01

## 8. Implementation Details

### 8.1 Software and Libraries
- **Programming Language:** Python 3.x
- **Data Manipulation:** pandas, NumPy
- **Deep Learning:** TensorFlow/Keras (or PyTorch)
- **NLP:** NLTK, spaCy, Transformers (Hugging Face)
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

### 8.2 Computational Resources
- Training performed on [specify: GPU model/cloud platform]
- Approximate training time per model: [to be determined]

### 8.3 Reproducibility
- Random seed set for reproducible results
- All hyperparameters documented
- Code version-controlled via GitHub

## 9. Current Progress

- ✅ Dataset acquisition (Lending Club via Kaggle)
- ✅ Initial data exploration (2.2M loans, 151 features)
- ✅ Loan status distribution analysis
- ✅ Feature identification and mapping
- 🔄 **In Progress:** Data preprocessing (temporal filtering)
- ⏳ **Pending:** Feature engineering
- ⏳ **Pending:** Text preprocessing pipeline
- ⏳ **Pending:** Model implementation and training
- ⏳ **Pending:** Performance evaluation and comparison

---

## References

[1] J. Kriebel and L. Stitz, "Credit default prediction from user-generated text in peer-to-peer lending using deep learning," *European Journal of Operational Research*, vol. 302, no. 1, pp. 309–323, 2022. DOI: 10.1016/j.ejor.2021.12.024

[2] "Lending Club Loan Data," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/wordsforthewise/lending-club. [Accessed: Dec. 1, 2025]

---

*Last Updated: December 2025*
*Status: Data Preprocessing Phase*