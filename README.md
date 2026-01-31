Clinical Diabetes Diagnosis: SVM vs. MLP Neural Network
This repository features a comparative analysis between Linear Support Vector Machines (SVM) and Multi-Layer Perceptron (MLP) neural networks to diagnose diabetes based on clinical data.

📊 Performance Summary
In recent testing with a real-world dataset, the MLP Classifier slightly outperformed the Linear SVM, demonstrating higher sensitivity (Recall) for diabetic cases.
Model	Accuracy	Recall (Class 1)	Key Hyperparameters
Linear SVM	60.94%	58%	C: 0.01
MLP Classifier	61.88%	67%	hidden_layer_sizes: (50,), solver: sgd
Note on Medical Sensitivity: In clinical diagnosis, Recall is critical. The MLP's 67% recall indicates it is significantly better at catching true diabetic cases compared to the SVM's 58%, reducing the risk of missing a diagnosis.

📂 Dataset Information
Total Records: 1,065 clinical samples.
Features: 9 columns including Gender, Age, BMI, OGTT (Glucose), and HbA1c.
Class Balance: Nearly 50/50 split (537 Non-Diabetic / 528 Diabetic), providing a robust foundation for training.

🧪 Model Details
1. Linear SVM
The SVM utilized a small regularization parameter (C=0.01), suggesting a high degree of overlap in the feature space, requiring a "softer" margin to prevent overfitting.
2. MLP (Neural Network)
The Multi-Layer Perceptron achieved its best performance using:
Architecture: One hidden layer with 50 neurons.
Activation: ReLU.
Optimizer: SGD (Stochastic Gradient Descent).
Optimization Note: The model required extensive training (2000+ iterations), indicating complex, non-linear relationships within the clinical features.

📈 Visualizations
The system automatically generates confusion matrices to visualize the trade-off between False Positives and False Negatives.
Confusion Matrix Comparison
Linear SVM	MLP Classifier
	
🛠️ Requirements & Setup
Dependencies: numpy, pandas, matplotlib, seaborn, scikit-learn, openpyxl.
Data: Ensure data_file.xlsx is in the root folder.
Execution:
code
Bash
python main.py

🔍 Observations
The results highlight that while linear models (SVM) provide a fast baseline, neural networks (MLP) are better suited for medical datasets where the boundary between "healthy" and "at-risk" is often non-linear and high-dimensional.
