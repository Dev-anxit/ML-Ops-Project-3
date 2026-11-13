# Student Performance Prediction (MLOps Project 3)

This repository contains an end-to-end Machine Learning pipeline predicting student academic performance (`Pass` / `Fail`) using a Decision Tree Classifier and Hyperparameter Tuning with `GridSearchCV`.

## 📁 Project Structure

```text
.
├── Student_Performance_DT.csv  # Student performance dataset
├── model3.py                   # Data preprocessing, model training, tuning, and evaluation script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Standard git ignore rules
└── README.md                   # Project documentation
```

## 🛠️ Features & Methodology

- **Data Preprocessing**:
  - Missing value inspection.
  - Encoding binary categorical variables (`Internet_Access`, `Extracurricular`, `Final_Result`).
- **Features Used**:
  - `Study_Hours`: Number of study hours per week
  - `Attendance`: Student class attendance rate (%)
  - `Previous_Score`: Previous examination score
  - `Assignments_Completed`: Number of completed assignments
  - `Sleep_Hours`: Average sleep hours per day
  - `Participation`: Classroom participation score
  - `Internet_Access`: Internet availability at home (Yes / No)
  - `Extracurricular`: Participation in extracurricular activities (Yes / No)
  - **Target**: `Final_Result` (Pass = 1, Fail = 0)
- **Modeling & Optimization**:
  - Baseline `DecisionTreeClassifier` (evaluated on an 80/20 train-test split).
  - Hyperparameter tuning using 5-Fold Cross-Validation with `GridSearchCV` over:
    - `criterion`: `['gini', 'entropy']`
    - `max_depth`: `[3, 5, 7, None]`
    - `min_samples_split`: `[2, 5, 10]`
    - `min_samples_leaf`: `[1, 3, 5]`

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Dev-anxit/ML-Ops-Project-3.git
cd ML-Ops-Project-3
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the model
```bash
python3 model3.py
```

## 📊 Results

- **Baseline Model**: ~67.00% test accuracy
- **Hyperparameter-Tuned Model**: ~69.00% test accuracy with 75.50% cross-validation score (`max_depth=5`, `min_samples_leaf=3`, `min_samples_split=10`)
