Diabetes Diagnosis System: A Comparative Machine Learning Approach
This project implements a clinical decision-support pipeline to detect diabetes using patient medical data. It compares the performance of a Linear Support Vector Machine (SVM) and a Multi-Layer Perceptron (MLP) Neural Network, focusing on hyperparameter tuning and medical evaluation metrics like Recall.

📌 Project Overview
The system processes clinical features (such as BMI, Glucose levels, and HbA1c) to classify patients as diabetic or non-diabetic. Unlike standard classification tasks, this project uses medical thresholds to dynamically label the dataset and employs GridSearchCV to optimize models for high-stakes medical diagnostics.

🧪 Clinical Features
The model utilizes the following patient features:
GENDER: Categorical (Encoded)
AGE: Patient age
BMI: Body Mass Index
OGTT1FBS: Oral Glucose Tolerance Test (Fast Blood Sugar)
HBA1C1: Hemoglobin A1c levels (Average blood sugar over 3 months)

⚙️ Methodology

1. Data Preprocessing & Medical Labeling
Data is cleaned of duplicates and labeled based on standard clinical diagnostic criteria:
Diabetic (1): If OGTT >= 126 mg/dL OR HbA1c >= 6.5%.
Non-Diabetic (0): Otherwise.
The features are then scaled using StandardScaler to ensure the Neural Network and SVM converge efficiently.

2. Model Architectures
Linear SVM: A robust boundary-based classifier. Optimized using different C (regularization) values.
MLP Classifier (Neural Network): A deep learning approach exploring different hidden layer architectures, activation functions (ReLU, Tanh), and solvers (Adam, SGD).

4. Hyperparameter Tuning
We use GridSearchCV with a focus on Recall. In a medical context, Recall is prioritized because missing a diabetic patient (False Negative) is more critical than a false alarm (False Positive).

📊 Results & Performance
Based on the latest run, the models achieved the following performance on a test set of 320 samples:
Metric	Linear SVM	MLP Classifier
Accuracy	60.94%	61.88%
Recall (Class 1)	0.58	0.67
Precision (Class 1)	0.61	0.61
Best Parameters	C: 0.01	hidden_layers: (50,), solver: sgd

Confusion Matrices
The project generates visual heatmaps to compare the error types between the two models:
Linear SVM: Shows a balanced approach but slightly higher False Negatives.
MLP Classifier: Achieved the highest accuracy and significantly better Recall for diabetic patients (detecting 106 out of 159 cases).

📈 Key Insights
Neural Networks vs. SVM: The MLP Classifier outperformed the Linear SVM in both overall accuracy and its ability to correctly identify diabetic patients (Recall).
Convergence: The MLP model encountered convergence warnings, suggesting that with a larger dataset or further feature engineering, the accuracy could be improved beyond the current ~62%.

📜 Future Improvements
Implement Random Forest or XGBoost for comparison.
Feature engineering to combine Age and BMI into risk-factor categories.
Addressing class imbalance if present in larger datasets using SMOTE.
